// Package mel is a prototype log-mel spectrogram generator that reproduces the
// BSG-BAT preprocessing (Zenodo 10.5281/zenodo.15495676) in Go, using the
// github.com/tphakala/simd kernels for the vectorizable inner loops.
//
// Pipeline (matches original_code/data384.py wav2spectrograms):
//
//	384 kHz mono PCM (float32, normalized to ~[-1,1])
//	-> frame (n_fft=1024, hop=768), periodic Hann window      [simd f32.Mul]
//	-> FFT 1024 -> power |.|^2 of first 513 bins              [fftPlan + simd c64.AbsSq]
//	-> Slaney mel filterbank (128 bins, 9-150 kHz)            [simd f32.DotProduct]
//	-> log10, per-512-frame normalize (mean/std, -median, clip 0..6)  [simd f32.Mean/StdDev]
//	-> model input tensor [512, 128]
//
// Framing uses center=false (streaming-friendly). librosa's default center=true
// adds reflect padding of n_fft/2; matching that bit-for-bit is a follow-up.
package mel

import (
	"math"
	"slices"

	"github.com/tphakala/go-spectrogram/internal/dsp"
	"github.com/tphakala/simd/c64"
	"github.com/tphakala/simd/f32"
)

const (
	SampleRate = 384000
	NFFT       = 1024
	HopLength  = 768
	NMels      = 128
	Fmin       = 9000.0
	Fmax       = 150000.0
	NFreq      = NFFT/2 + 1 // 513
	SegFrames  = 512        // model input time dimension
	SegHop     = 250        // 0.5 s at hop 768 / 384 kHz
)

// Generator holds the precomputed window, mel filterbank, FFT plan, and scratch
// buffers. It is NOT safe for concurrent use (reuses scratch); make one per
// worker goroutine. Construction allocates; the hot path does not.
type Generator struct {
	window []float32   // periodic Hann, len NFFT
	fb     [][]float32 // mel filterbank, NMels x NFreq (dense rows for simd dot)
	plan   *dsp.FFTPlan

	frame []float32   // windowed frame scratch, len NFFT
	cin   []complex64 // complex input scratch, len NFFT
	power []float32   // power spectrum scratch, len NFreq
}

// NewGenerator builds a generator for the fixed BSG-BAT parameters.
func NewGenerator() *Generator {
	return &Generator{
		window: hannPeriodic(NFFT),
		fb:     melFilterbankSlaney(),
		plan:   dsp.NewFFTPlan(NFFT),
		frame:  make([]float32, NFFT),
		cin:    make([]complex64, NFFT),
		power:  make([]float32, NFreq),
	}
}

// NumFrames reports how many hop-spaced frames a signal of n samples yields.
func NumFrames(n int) int {
	if n < NFFT {
		return 0
	}
	return 1 + (n-NFFT)/HopLength
}

// MelPower writes the (un-logged) mel power spectrogram for signal into dst,
// row-major [nFrames][NMels]. dst must have length NumFrames(len(signal))*NMels.
// Returns the number of frames written. This is the hot path.
func (g *Generator) MelPower(signal []float32, dst []float32) int {
	nFrames := NumFrames(len(signal))
	for fr := 0; fr < nFrames; fr++ {
		off := fr * HopLength
		// window: frame = signal[off:off+NFFT] .* window      (simd)
		f32.Mul(g.frame, signal[off:off+NFFT], g.window)
		// real -> complex, FFT, power of first NFreq bins      (simd for FromReal/AbsSq)
		c64.FromReal(g.cin, g.frame)
		spec := g.plan.Forward(g.cin)
		c64.AbsSq(g.power, spec[:NFreq])
		// mel projection: 128 dense dot products of length 513  (simd)
		row := dst[fr*NMels : (fr+1)*NMels]
		for i := 0; i < NMels; i++ {
			row[i] = f32.DotProduct(g.power, g.fb[i])
		}
	}
	return nFrames
}

// MelSpectrogram allocates and returns the mel power spectrogram (row-major)
// plus its frame count. Convenience wrapper over MelPower.
func (g *Generator) MelSpectrogram(signal []float32) (data []float32, nFrames int) {
	nFrames = NumFrames(len(signal))
	data = make([]float32, nFrames*NMels)
	g.MelPower(signal, data)
	return
}

// NormalizeSegment turns one SegFrames x NMels block of mel POWER values into
// the model input: log10(x+1e-6), then (x-mean)/std over the whole block, then
// subtract the per-mel-bin median over time and clip to [0,6]. In place.
// seg is row-major [SegFrames][NMels] (len SegFrames*NMels).
func NormalizeSegment(seg []float32) {
	for i := range seg {
		seg[i] = float32(math.Log10(float64(seg[i]) + 1e-6))
	}
	mean := f32.Mean(seg)
	std := f32.StdDev(seg)
	if std == 0 {
		std = 1
	}
	inv := 1.0 / std
	for i := range seg {
		seg[i] = (seg[i] - mean) * inv
	}
	col := make([]float32, SegFrames) // single reusable scratch for all bins
	for b := 0; b < NMels; b++ {
		for t := 0; t < SegFrames; t++ {
			col[t] = seg[t*NMels+b]
		}
		med := medianInPlace(col) // sorts col; original values already copied out
		for t := 0; t < SegFrames; t++ {
			v := seg[t*NMels+b] - med
			switch {
			case v < 0:
				v = 0
			case v > 6:
				v = 6
			}
			seg[t*NMels+b] = v
		}
	}
}

// medianInPlace sorts a and returns its median. a is scratch the caller owns.
func medianInPlace(a []float32) float32 {
	slices.Sort(a)
	n := len(a)
	if n%2 == 1 {
		return a[n/2]
	}
	return 0.5 * (a[n/2-1] + a[n/2])
}

// hannPeriodic returns a periodic Hann window (librosa fftbins=True):
// w[n] = 0.5 - 0.5*cos(2*pi*n/N).
func hannPeriodic(n int) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n)))
	}
	return w
}

// --- Slaney mel filterbank (librosa default: htk=false, norm="slaney") ---

func hzToMel(f float64) float64 {
	const fsp = 200.0 / 3.0
	const minLogHz = 1000.0
	minLogMel := minLogHz / fsp
	logstep := math.Log(6.4) / 27.0
	if f >= minLogHz {
		return minLogMel + math.Log(f/minLogHz)/logstep
	}
	return f / fsp
}

func melToHz(m float64) float64 {
	const fsp = 200.0 / 3.0
	const minLogHz = 1000.0
	minLogMel := minLogHz / fsp
	logstep := math.Log(6.4) / 27.0
	if m >= minLogMel {
		return minLogHz * math.Exp(logstep*(m-minLogMel))
	}
	return fsp * m
}

func melFilterbankSlaney() [][]float32 {
	fftfreqs := make([]float64, NFreq)
	for i := range fftfreqs {
		fftfreqs[i] = float64(i) * (SampleRate / 2.0) / float64(NFFT/2)
	}
	mlo, mhi := hzToMel(Fmin), hzToMel(Fmax)
	melPts := make([]float64, NMels+2)
	for i := range melPts {
		melPts[i] = melToHz(mlo + (mhi-mlo)*float64(i)/float64(NMels+1))
	}
	fb := make([][]float32, NMels)
	for i := 0; i < NMels; i++ {
		fb[i] = make([]float32, NFreq)
		fdiffLo := melPts[i+1] - melPts[i]
		fdiffHi := melPts[i+2] - melPts[i+1]
		enorm := 2.0 / (melPts[i+2] - melPts[i])
		for k := 0; k < NFreq; k++ {
			lower := (fftfreqs[k] - melPts[i]) / fdiffLo
			upper := (melPts[i+2] - fftfreqs[k]) / fdiffHi
			w := math.Min(lower, upper)
			if w < 0 {
				w = 0
			}
			fb[i][k] = float32(w * enorm)
		}
	}
	return fb
}
