package fft

import (
	"math"
	"testing"

	"github.com/tphakala/simd/f64"
)

// dftSizes are the transform sizes the sox renderer actually uses (YSize
// 129/257/513/1025 -> dft 256/512/1024/2048), plus the small sizes that
// exercise the stage-count edge cases.
var dftSizes = []int{4, 8, 16, 32, 64, 256, 512, 1024, 2048}

// hann returns a symmetric Hann window of length n, matching what the sox
// analyzer feeds in (makeWindow builds one of length dft+1 and only the first
// dft entries are read).
func hann(n int) []float64 {
	w := make([]float64, n)
	m := float64(n - 1)
	for i := range w {
		w[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/m)
	}
	return w
}

func testSignal(n int) []float64 {
	s := make([]float64, n)
	for i := range s {
		t := float64(i) / 24000
		s[i] = 0.4*math.Sin(2*math.Pi*1000*t) +
			0.2*math.Sin(2*math.Pi*3500*t) +
			0.1*math.Sin(2*math.Pi*7000*t)
	}
	return s
}

// referencePower computes the same one-frame power spectrum via simd's
// f64.STFTPlan, which is the implementation this package replaces and which is
// already known to render bit-exactly against the SoX binary.
func referencePower(t *testing.T, nfft int, signal, window []float64) []float64 {
	t.Helper()
	ref, err := f64.NewSTFTPlan(nfft)
	if err != nil {
		t.Fatalf("simd plan: %v", err)
	}
	dst := make([]float64, nfft/2+1)
	if got := ref.STFTPowerInto(dst, signal, window, nfft, f64.NoPad); got != 1 {
		t.Fatalf("simd reference wrote %d frames, want 1", got)
	}
	return dst
}

// TestPowerIntoMatchesSIMD is the acceptance test for the vendored transform:
// it must agree with simd's STFTPowerInto closely enough that no palette index
// can move. A radix-4 decomposition reorders the arithmetic, so the results are
// not required to be bit-identical, only to agree to a relative error far below
// what a 0.775 dB palette step could notice.
func TestPowerIntoMatchesSIMD(t *testing.T) {
	const tol = 1e-12
	for _, nfft := range dftSizes {
		t.Run(sizeName(nfft), func(t *testing.T) {
			sig := testSignal(nfft)
			win := hann(nfft)
			want := referencePower(t, nfft, sig, win)

			p, err := NewPlan(nfft)
			if err != nil {
				t.Fatalf("NewPlan: %v", err)
			}
			if p.NumBins() != nfft/2+1 {
				t.Fatalf("NumBins = %d, want %d", p.NumBins(), nfft/2+1)
			}
			got := make([]float64, p.NumBins())
			p.PowerInto(got, sig, win)

			// Scale the comparison by the spectrum peak: bins in a deep null
			// carry no energy and their relative error is meaningless.
			peak := 0.0
			for _, v := range want {
				peak = math.Max(peak, v)
			}
			for k := range want {
				if diff := math.Abs(got[k] - want[k]); diff > tol*peak {
					t.Fatalf("bin %d: got %g, want %g (diff %g > %g)",
						k, got[k], want[k], diff, tol*peak)
				}
			}
		})
	}
}

// TestPowerIntoRectangularWindow covers the nil-window path.
func TestPowerIntoRectangularWindow(t *testing.T) {
	const nfft = 256
	sig := testSignal(nfft)
	want := referencePower(t, nfft, sig, nil)

	p, err := NewPlan(nfft)
	if err != nil {
		t.Fatalf("NewPlan: %v", err)
	}
	got := make([]float64, p.NumBins())
	p.PowerInto(got, sig, nil)

	peak := 0.0
	for _, v := range want {
		peak = math.Max(peak, v)
	}
	for k := range want {
		if diff := math.Abs(got[k] - want[k]); diff > 1e-12*peak {
			t.Fatalf("bin %d: got %g, want %g", k, got[k], want[k])
		}
	}
}

// TestPowerIntoPureTone checks the transform independently of the reference:
// a bin-centred sinusoid must put essentially all its energy in that one bin.
func TestPowerIntoPureTone(t *testing.T) {
	const nfft = 1024
	const bin = 64
	sig := make([]float64, nfft)
	for i := range sig {
		sig[i] = math.Sin(2 * math.Pi * float64(bin) * float64(i) / float64(nfft))
	}
	p, err := NewPlan(nfft)
	if err != nil {
		t.Fatalf("NewPlan: %v", err)
	}
	got := make([]float64, p.NumBins())
	p.PowerInto(got, sig, nil)

	peakBin, peak := 0, 0.0
	for k, v := range got {
		if v > peak {
			peak, peakBin = v, k
		}
	}
	if peakBin != bin {
		t.Fatalf("peak at bin %d, want %d", peakBin, bin)
	}
	// Every other bin should be orders of magnitude down.
	for k, v := range got {
		if k != bin && v > peak*1e-12 {
			t.Errorf("bin %d has %g, want << peak %g", k, v, peak)
		}
	}
}

func TestNewPlanRejectsNonPowerOfTwo(t *testing.T) {
	for _, n := range []int{0, 1, 3, 100, 1000} {
		if _, err := NewPlan(n); err == nil {
			t.Errorf("NewPlan(%d): expected an error, got nil", n)
		}
	}
}

func sizeName(n int) string {
	switch n {
	case 4:
		return "nfft4"
	case 8:
		return "nfft8"
	case 16:
		return "nfft16"
	case 32:
		return "nfft32"
	case 64:
		return "nfft64"
	case 256:
		return "nfft256"
	case 512:
		return "nfft512"
	case 1024:
		return "nfft1024"
	case 2048:
		return "nfft2048"
	}
	return "nfft?"
}

// TestPowerIntoWritesEveryBin guards the paired unravel: bins are emitted two
// at a time with DC, Nyquist and the self-paired middle bin special-cased, so a
// bin could be skipped without any value comparison noticing. Pre-filling with
// NaN makes an unwritten bin impossible to miss.
func TestPowerIntoWritesEveryBin(t *testing.T) {
	for nfft := 4; nfft <= 4096; nfft <<= 1 {
		p, err := NewPlan(nfft)
		if err != nil {
			t.Fatalf("NewPlan(%d): %v", nfft, err)
		}
		dst := make([]float64, p.NumBins())
		for i := range dst {
			dst[i] = math.NaN()
		}
		p.PowerInto(dst, testSignal(nfft), hann(nfft))
		for k, v := range dst {
			if math.IsNaN(v) {
				t.Errorf("nfft=%d: bin %d was never written", nfft, k)
			}
		}
	}
}

// TestPowerIntoAllSizes runs the reference comparison across every power of two
// the plan supports in the useful range, so both radix decompositions
// ([4,4,...] when log2(half) is even, [2,4,4,...] when odd) are covered at many
// stage counts rather than only at the four sizes sox happens to use.
func TestPowerIntoAllSizes(t *testing.T) {
	for nfft := 4; nfft <= 4096; nfft <<= 1 {
		sig := testSignal(nfft)
		win := hann(nfft)
		want := referencePower(t, nfft, sig, win)

		p, err := NewPlan(nfft)
		if err != nil {
			t.Fatalf("NewPlan(%d): %v", nfft, err)
		}
		got := make([]float64, p.NumBins())
		p.PowerInto(got, sig, win)

		peak := 0.0
		for _, v := range want {
			peak = math.Max(peak, v)
		}
		for k := range want {
			if diff := math.Abs(got[k] - want[k]); diff > 1e-12*peak {
				t.Fatalf("nfft=%d bin %d: got %g, want %g (diff %g)",
					nfft, k, got[k], want[k], diff)
			}
		}
	}
}

// TestPowerIntoAllocFree keeps the transform off the heap: it runs once per
// block for every column of every render.
func TestPowerIntoAllocFree(t *testing.T) {
	p, err := NewPlan(1024)
	if err != nil {
		t.Fatal(err)
	}
	sig, win := testSignal(1024), hann(1024)
	dst := make([]float64, p.NumBins())
	if n := testing.AllocsPerRun(100, func() { p.PowerInto(dst, sig, win) }); n != 0 {
		t.Errorf("PowerInto allocates %v times per call, want 0", n)
	}
}
