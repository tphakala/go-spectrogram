package fft

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"sync"
	"testing"

	"github.com/tphakala/simd/f64"
)

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
// f64.STFTPlan, the implementation this package was measured against.
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

// naivePower is an independent O(n^2) DFT, written straight from the definition
// with no shared code path. The differential tests all compare against simd, so
// a bug in simd would be agreed with unanimously; this is the only ground truth
// in the package that does not depend on another FFT.
func naivePower(signal, window []float64, nfft int) []float64 {
	x := make([]complex128, nfft)
	for n := range x {
		v := signal[n]
		if window != nil {
			v *= window[n]
		}
		x[n] = complex(v, 0)
	}
	out := make([]float64, nfft/2+1)
	for k := range out {
		var acc complex128
		for n := 0; n < nfft; n++ {
			ang := -2 * math.Pi * float64(k) * float64(n) / float64(nfft)
			acc += x[n] * cmplx.Exp(complex(0, ang))
		}
		out[k] = real(acc)*real(acc) + imag(acc)*imag(acc)
	}
	return out
}

// assertClose compares a power spectrum against a reference with a hybrid
// tolerance.
//
// A purely peak-relative bound is close to vacuous: at nfft=2048, 68% of bins
// sit more than 1e-12 below the peak, so their value is unconstrained and a
// transform that flushed them to zero would still pass. The per-bin term is
// what actually constrains those bins; the peak term keeps deep nulls, where
// relative error is meaningless, from failing on rounding noise.
func assertClose(t *testing.T, got, want []float64, atolPeak, rtol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d, want %d", len(got), len(want))
	}
	peak := 0.0
	for _, v := range want {
		peak = math.Max(peak, v)
	}
	for k := range want {
		tol := atolPeak*peak + rtol*math.Abs(want[k])
		if diff := math.Abs(got[k] - want[k]); diff > tol {
			t.Fatalf("bin %d: got %g, want %g (diff %g > tol %g)", k, got[k], want[k], diff, tol)
		}
	}
}

// TestPowerIntoMatchesSIMD is the acceptance test: the vendored transform must
// agree with simd's across every size the plan supports, so both decompositions
// ([4,4,...] when log2(half) is even, [2,4,4,...] when odd) are covered at many
// stage counts rather than only the four sizes sox uses.
func TestPowerIntoMatchesSIMD(t *testing.T) {
	for nfft := 4; nfft <= 4096; nfft <<= 1 {
		t.Run(fmt.Sprintf("nfft%d", nfft), func(t *testing.T) {
			sig, win := testSignal(nfft), hann(nfft)
			want := referencePower(t, nfft, sig, win)

			p, err := NewPlan(nfft)
			if err != nil {
				t.Fatalf("NewPlan: %v", err)
			}
			if p.NumBins() != nfft/2+1 || p.NFFT() != nfft {
				t.Fatalf("NumBins/NFFT = %d/%d, want %d/%d", p.NumBins(), p.NFFT(), nfft/2+1, nfft)
			}
			got := make([]float64, p.NumBins())
			p.PowerInto(got, sig, win)
			assertClose(t, got, want, 1e-14, 1e-9)
		})
	}
}

// TestPowerIntoMatchesNaiveDFT is the reference-independent check. It is
// quadratic, so it runs only at small sizes, but it covers both radix
// decompositions and is the only assertion here that would survive simd itself
// being wrong.
func TestPowerIntoMatchesNaiveDFT(t *testing.T) {
	for nfft := 4; nfft <= 256; nfft <<= 1 {
		t.Run(fmt.Sprintf("nfft%d", nfft), func(t *testing.T) {
			for _, win := range [][]float64{nil, hann(nfft)} {
				sig := testSignal(nfft)
				want := naivePower(sig, win, nfft)
				p, err := NewPlan(nfft)
				if err != nil {
					t.Fatalf("NewPlan: %v", err)
				}
				got := make([]float64, p.NumBins())
				p.PowerInto(got, sig, win)
				assertClose(t, got, want, 1e-12, 1e-6)
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
	assertClose(t, got, want, 1e-14, 1e-9)
}

// TestPowerIntoPlanReuse covers the way the renderer actually calls this: one
// plan, thousands of frames. Every other value test builds a fresh plan and
// calls once, so stale scratch surviving between calls (a non-bijective
// permutation, say) would go unnoticed.
func TestPowerIntoPlanReuse(t *testing.T) {
	const nfft = 1024
	p, err := NewPlan(nfft)
	if err != nil {
		t.Fatalf("NewPlan: %v", err)
	}
	win := hann(nfft)
	signals := [][]float64{
		testSignal(nfft),
		make([]float64, nfft), // silence between two live frames
		hann(nfft),            // a different shape entirely
		testSignal(nfft),      // the first signal again: must reproduce exactly
	}
	var first []float64
	for i, sig := range signals {
		got := make([]float64, p.NumBins())
		p.PowerInto(got, sig, win)

		fresh, err := NewPlan(nfft)
		if err != nil {
			t.Fatalf("NewPlan: %v", err)
		}
		want := make([]float64, fresh.NumBins())
		fresh.PowerInto(want, sig, win)

		for k := range want {
			if got[k] != want[k] {
				t.Fatalf("frame %d bin %d: reused plan gave %g, fresh plan %g", i, k, got[k], want[k])
			}
		}
		if i == 0 {
			first = got
		}
		if i == len(signals)-1 {
			for k := range first {
				if got[k] != first[k] {
					t.Fatalf("bin %d: repeat of frame 0 gave %g, want %g", k, got[k], first[k])
				}
			}
		}
	}
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

// TestPowerIntoPureTone checks the transform independently of any reference: a
// bin-centred sinusoid must put essentially all its energy in that one bin. The
// bound is tight deliberately; the achieved off-bin leakage is ~1e-29 of the
// peak, so a loose bound here would pass a badly smeared transform.
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
	for k, v := range got {
		if k != bin && v > peak*1e-20 {
			t.Errorf("bin %d has %g, want << peak %g", k, v, peak)
		}
	}
}

// TestPowerIntoParseval asserts energy conservation, a global invariant that no
// per-bin comparison against another FFT can give: it holds independently of
// any reference implementation and constrains the low-energy bins that a
// peak-relative tolerance leaves free.
func TestPowerIntoParseval(t *testing.T) {
	for nfft := 4; nfft <= 2048; nfft <<= 1 {
		p, err := NewPlan(nfft)
		if err != nil {
			t.Fatalf("NewPlan(%d): %v", nfft, err)
		}
		sig := testSignal(nfft)
		got := make([]float64, p.NumBins())
		p.PowerInto(got, sig, nil)

		var time, freq float64
		for _, v := range sig {
			time += v * v
		}
		// The half-spectrum counts every interior bin once; its mirror supplies
		// the other half of the energy.
		freq = got[0] + got[p.NumBins()-1]
		for k := 1; k < p.NumBins()-1; k++ {
			freq += 2 * got[k]
		}
		freq /= float64(nfft)
		if rel := math.Abs(freq-time) / math.Max(time, 1e-300); rel > 1e-12 {
			t.Errorf("nfft=%d: Parseval mismatch, time %g vs freq %g (rel %g)", nfft, time, freq, rel)
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

func TestNewPlanRejectsUnsupportedSizes(t *testing.T) {
	// 2 is the interesting one: it IS a power of two and is still rejected,
	// which is why the sentinel is not named for that condition alone.
	for _, n := range []int{-4, 0, 1, 2, 3, 100, 1000} {
		_, err := NewPlan(n)
		if err == nil {
			t.Errorf("NewPlan(%d): expected an error, got nil", n)
			continue
		}
		if !errors.Is(err, ErrUnsupportedSize) {
			t.Errorf("NewPlan(%d): got %v, want ErrUnsupportedSize", n, err)
		}
	}
}

// TestPowerIntoRejectsShortSlices covers the contract that a slice expression
// alone does not enforce: re-slicing bound-checks against capacity, so a short
// slice with spare capacity would otherwise be accepted and silently read past
// its length.
func TestPowerIntoRejectsShortSlices(t *testing.T) {
	const nfft = 64
	p, err := NewPlan(nfft)
	if err != nil {
		t.Fatal(err)
	}
	full := make([]float64, nfft)
	win := hann(nfft)
	bins := make([]float64, p.NumBins())

	cases := []struct {
		name                string
		dst, signal, window []float64
	}{
		// Each short slice keeps the full backing array, so cap is large and
		// only the explicit length check can catch it.
		{"short signal with spare cap", bins, full[:nfft/2], win},
		{"short window with spare cap", bins, full, win[:nfft/2]},
		{"short dst with spare cap", bins[:2], full, win},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected a panic, got none")
				}
			}()
			p.PowerInto(c.dst, c.signal, c.window)
		})
	}
}

// FuzzPowerInto asserts the invariants that hold for every in-contract input:
// no panic, and a power spectrum that is non-negative everywhere. Example
// tables cover the sizes and one signal shape; this covers the value domain,
// including the non-finite inputs the renderer can be fed.
func FuzzPowerInto(f *testing.F) {
	f.Add(2, uint64(1), false)
	f.Add(5, uint64(0xdeadbeef), true)
	f.Add(0, uint64(7), true)

	f.Fuzz(func(t *testing.T, sizeIdx int, seed uint64, useWindow bool) {
		if sizeIdx < 0 || sizeIdx > 8 {
			t.Skip()
		}
		nfft := 4 << uint(sizeIdx)
		p, err := NewPlan(nfft)
		if err != nil {
			t.Fatalf("NewPlan(%d): %v", nfft, err)
		}

		// A cheap xorshift keeps the corpus small while still sweeping the
		// value domain, including the non-finite values sox can encounter.
		sig := make([]float64, nfft)
		s := seed | 1
		for i := range sig {
			s ^= s << 13
			s ^= s >> 7
			s ^= s << 17
			switch s % 32 {
			case 0:
				sig[i] = math.NaN()
			case 1:
				sig[i] = math.Inf(1)
			case 2:
				sig[i] = math.Inf(-1)
			default:
				sig[i] = float64(int64(s%2001)-1000) / 1000
			}
		}
		var win []float64
		if useWindow {
			win = hann(nfft)
		}

		dst := make([]float64, p.NumBins())
		p.PowerInto(dst, sig, win) // must not panic

		// Non-negativity would be unfalsifiable here: every bin is written as
		// a*a + b*b. Assert properties that a wrong transform can actually
		// violate instead.
		allFinite := true
		for _, v := range sig {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				allFinite = false
				break
			}
		}
		if !allFinite {
			return // NaN/Inf legitimately propagate; only the no-panic contract applies
		}

		for k, v := range dst {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("nfft=%d bin %d: finite input produced %g", nfft, k, v)
			}
		}
		if win != nil {
			return // Parseval below assumes an unwindowed frame
		}
		// Parseval: the half-spectrum carries the same energy as the samples,
		// counting interior bins twice for their mirrored partners. This is a
		// global invariant, so unlike a per-bin comparison it constrains the
		// low-energy bins a peak-relative tolerance leaves free.
		var timeE float64
		for _, v := range sig {
			timeE += v * v
		}
		freqE := dst[0] + dst[len(dst)-1]
		for k := 1; k < len(dst)-1; k++ {
			freqE += 2 * dst[k]
		}
		freqE /= float64(nfft)
		if timeE > 1e-9 {
			if rel := math.Abs(freqE-timeE) / timeE; rel > 1e-9 {
				t.Fatalf("nfft=%d: Parseval violated, time %g vs freq %g (rel %g)", nfft, timeE, freqE, rel)
			}
		}
	})
}

// TestCloneMatchesOriginal pins Clone's contract: a clone shares the parent's
// twiddle tables but carries its own scratch, so it must produce bit-identical
// output. Bit-identical rather than approximately equal, because the renderer's
// parity with the sox binary is exact and a clone is what every worker beyond
// the first one uses.
func TestCloneMatchesOriginal(t *testing.T) {
	for nfft := 4; nfft <= 2048; nfft <<= 1 {
		p, err := NewPlan(nfft)
		if err != nil {
			t.Fatalf("nfft %d: %v", nfft, err)
		}
		sig := make([]float64, nfft)
		win := make([]float64, nfft)
		for i := range sig {
			sig[i] = math.Sin(2*math.Pi*3*float64(i)/float64(nfft)) + 0.25*math.Cos(float64(i))
			win[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(nfft-1))
		}
		want := make([]float64, p.NumBins())
		p.PowerInto(want, sig, win)

		c := p.Clone()
		got := make([]float64, c.NumBins())
		c.PowerInto(got, sig, win)

		if c.NFFT() != p.NFFT() || c.NumBins() != p.NumBins() {
			t.Fatalf("nfft %d: clone reports size %d/%d, original %d/%d",
				nfft, c.NFFT(), c.NumBins(), p.NFFT(), p.NumBins())
		}
		for k := range want {
			if got[k] != want[k] {
				t.Fatalf("nfft %d bin %d: clone %v, original %v (must be bit-identical)",
					nfft, k, got[k], want[k])
			}
		}
		// Interleaving the two plans must not disturb either. This is a
		// weaker check than it looks: PowerInto repacks the whole scratch on
		// every call, so a clone that wrongly SHARED the scratch would still
		// pass here. Sequential use cannot distinguish the two; only
		// concurrent use can, which is what TestCloneIsRaceFree covers.
		again := make([]float64, p.NumBins())
		p.PowerInto(again, sig, win)
		for k := range want {
			if again[k] != want[k] {
				t.Fatalf("nfft %d bin %d: original returned %v on a repeat run, first run gave %v",
					nfft, k, again[k], want[k])
			}
		}
	}
}

// TestCloneIsRaceFree runs a plan and several clones concurrently over the same
// read-only inputs. It exists to be run under -race: sharing the twiddle tables
// is the whole point of Clone, and a table that turned out to be written during
// a transform would be a data race in every parallel render.
func TestCloneIsRaceFree(t *testing.T) {
	const nfft = 512
	p, err := NewPlan(nfft)
	if err != nil {
		t.Fatal(err)
	}
	sig := make([]float64, nfft)
	win := make([]float64, nfft)
	for i := range sig {
		sig[i] = math.Sin(float64(i))
		win[i] = 1
	}
	want := make([]float64, p.NumBins())
	p.PowerInto(want, sig, win)

	plans := []*Plan{p}
	for range 7 {
		plans = append(plans, p.Clone())
	}
	var wg sync.WaitGroup
	errs := make([]bool, len(plans))
	for i, pl := range plans {
		wg.Add(1)
		go func() {
			defer wg.Done()
			dst := make([]float64, pl.NumBins())
			for range 50 {
				pl.PowerInto(dst, sig, win)
				for k := range want {
					if dst[k] != want[k] {
						errs[i] = true
						return
					}
				}
			}
		}()
	}
	wg.Wait()
	for i, bad := range errs {
		if bad {
			t.Errorf("plan %d produced a different spectrum when run concurrently", i)
		}
	}
}
