package mel

import (
	"math"
	"math/rand"
	"testing"
)

// TestFFTTone proves the FFT is correct: a pure cosine at bin k0 must put almost
// all energy in bin k0 (and its mirror), with a matching DFT cross-check.
func TestFFTTone(t *testing.T) {
	p := newFFTPlan(NFFT)
	const k0 = 37
	in := make([]complex64, NFFT)
	for n := 0; n < NFFT; n++ {
		in[n] = complex64(complex(math.Cos(2*math.Pi*float64(k0)*float64(n)/float64(NFFT)), 0))
	}
	out := p.forward(in)

	peak, peakMag := -1, float64(-1)
	for k := 0; k < NFFT/2+1; k++ {
		m := float64(real(out[k]))*float64(real(out[k])) + float64(imag(out[k]))*float64(imag(out[k]))
		if m > peakMag {
			peakMag, peak = m, k
		}
	}
	if peak != k0 {
		t.Fatalf("FFT peak at bin %d, want %d", peak, k0)
	}
	// Naive DFT cross-check at the peak bin.
	var re, im float64
	for n := 0; n < NFFT; n++ {
		ang := -2 * math.Pi * float64(k0) * float64(n) / float64(NFFT)
		re += math.Cos(2*math.Pi*float64(k0)*float64(n)/float64(NFFT)) * math.Cos(ang)
		im += math.Cos(2*math.Pi*float64(k0)*float64(n)/float64(NFFT)) * math.Sin(ang)
	}
	got := math.Hypot(float64(real(out[k0])), float64(imag(out[k0])))
	want := math.Hypot(re, im)
	if math.Abs(got-want)/want > 1e-3 {
		t.Fatalf("FFT magnitude at bin %d = %.3f, DFT says %.3f", k0, got, want)
	}
}

func TestMelShape(t *testing.T) {
	g := NewGenerator()
	sig := make([]float32, SampleRate) // 1 s
	data, nf := g.MelSpectrogram(sig)
	wantFrames := 1 + (SampleRate-NFFT)/HopLength
	if nf != wantFrames {
		t.Fatalf("frames = %d, want %d", nf, wantFrames)
	}
	if len(data) != nf*NMels {
		t.Fatalf("len(data) = %d, want %d", len(data), nf*NMels)
	}
}

// TestNormalizeRange checks the normalize output stays in [0,6] and is finite.
func TestNormalizeRange(t *testing.T) {
	g := NewGenerator()
	rng := rand.New(rand.NewSource(1))
	sig := make([]float32, SegFrames*HopLength+NFFT)
	for i := range sig {
		sig[i] = float32(rng.NormFloat64() * 0.1)
	}
	data, nf := g.MelSpectrogram(sig)
	if nf < SegFrames {
		t.Fatalf("not enough frames: %d", nf)
	}
	seg := make([]float32, SegFrames*NMels)
	copy(seg, data[:SegFrames*NMels])
	NormalizeSegment(seg)
	for i, v := range seg {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("seg[%d] not finite: %v", i, v)
		}
		if v < 0 || v > 6 {
			t.Fatalf("seg[%d] = %v out of [0,6]", i, v)
		}
	}
}

// BenchmarkMelPower1s: full mel power spectrogram for 1 second of 384 kHz audio.
func BenchmarkMelPower1s(b *testing.B) {
	g := NewGenerator()
	sig := make([]float32, SampleRate)
	rng := rand.New(rand.NewSource(2))
	for i := range sig {
		sig[i] = float32(rng.NormFloat64() * 0.1)
	}
	dst := make([]float32, NumFrames(len(sig))*NMels)
	b.SetBytes(int64(len(sig) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.MelPower(sig, dst)
	}
}

// BenchmarkSegmentInference: mel + normalize for ONE 512-frame model window
// (~1.024 s of audio), i.e. the work per inference window.
func BenchmarkSegmentInference(b *testing.B) {
	g := NewGenerator()
	n := SegFrames*HopLength + NFFT
	sig := make([]float32, n)
	rng := rand.New(rand.NewSource(3))
	for i := range sig {
		sig[i] = float32(rng.NormFloat64() * 0.1)
	}
	dst := make([]float32, NumFrames(len(sig))*NMels)
	seg := make([]float32, SegFrames*NMels)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.MelPower(sig, dst)
		copy(seg, dst[:SegFrames*NMels])
		NormalizeSegment(seg)
	}
}
