package sox

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

// parallelCases covers the geometries whose scheduling differs: one DFT per
// column and several, an image that fills its xSize and one that truncates, a
// clip shorter than a single frame, and the -n path whose autogain is a
// reduction across workers.
func parallelCases() []struct {
	name string
	opt  Options
	n    int
	rate float64
} {
	return []struct {
		name string
		opt  Options
		n    int
		rate float64
	}{
		{"one-dft-per-column", Options{XSize: 1026, YSize: 513}, 15 * 24000, 24000},
		{"many-dfts-per-column", Options{XSize: 258, YSize: 129}, 15 * 24000, 24000},
		{"raw", Options{XSize: 514, YSize: 257, Raw: true}, 5 * 22050, 22050},
		{"normalize", Options{XSize: 514, YSize: 257, Normalize: true}, 5 * 22050, 22050},
		{"truncated", Options{XSize: 100, YSize: 129, PixelsPerSec: 500}, 3 * 16000, 16000},
		{"short-clip", Options{XSize: 258, YSize: 513}, 700, 8000},
		{"empty-clip", Options{XSize: 258, YSize: 129}, 0, 8000},
		{"odd-length", Options{XSize: 300, YSize: 129}, 44101, 44100},
		{"gain-and-range", Options{XSize: 258, YSize: 129, Gain: -20, DBRange: 65}, 4 * 8000, 8000},
	}
}

func parallelSignal(n int) []float32 {
	rng := rand.New(rand.NewSource(7))
	s := make([]float32, n)
	for i := range s {
		t := float64(i) / 24000
		// Tones over a noise floor: without the noise most cells land below the
		// dB floor, which would hide any divergence in the quantisation.
		s[i] = float32(0.4*math.Sin(2*math.Pi*997*t) +
			0.1*math.Sin(2*math.Pi*4001*t) +
			0.002*rng.NormFloat64())
	}
	return s
}

// TestParallelMatchesSequential is the invariant the whole schedule/execute
// split exists to protect: how many goroutines the columns are spread over must
// not be observable in the output. Worker counts that do not divide the column
// count are included on purpose, since those are what put a column boundary
// mid-window-reshape.
func TestParallelMatchesSequential(t *testing.T) {
	for _, tc := range parallelCases() {
		t.Run(tc.name, func(t *testing.T) {
			sig := parallelSignal(tc.n)
			seqOpt := tc.opt
			seqOpt.Workers = 1
			want, err := Render(sig, tc.rate, seqOpt)
			if err != nil {
				t.Fatalf("sequential render: %v", err)
			}
			for _, w := range []int{2, 3, 4, 5, 8, 17, 64, 4096} {
				opt := tc.opt
				opt.Workers = w
				got, err := Render(sig, tc.rate, opt)
				if err != nil {
					t.Fatalf("workers=%d: %v", w, err)
				}
				if got.Bounds() != want.Bounds() {
					t.Fatalf("workers=%d: bounds %v, sequential %v", w, got.Bounds(), want.Bounds())
				}
				if !bytes.Equal(got.Pix, want.Pix) {
					n := 0
					for i := range got.Pix {
						if got.Pix[i] != want.Pix[i] {
							n++
						}
					}
					t.Fatalf("workers=%d: %d of %d pixels differ from the sequential render",
						w, n, len(want.Pix))
				}
			}
		})
	}
}

// TestWorkersDefaultMatchesExplicit pins that leaving Workers unset renders the
// same image as any explicit setting, so the default is a latency choice only.
func TestWorkersDefaultMatchesExplicit(t *testing.T) {
	sig := parallelSignal(10 * 24000)
	opt := Options{XSize: 514, YSize: 257}
	want, err := Render(sig, 24000, opt)
	if err != nil {
		t.Fatal(err)
	}
	seq := opt
	seq.Workers = 1
	got, err := Render(sig, 24000, seq)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got.Pix, want.Pix) {
		t.Error("the default worker count renders a different image than Workers=1")
	}
}
