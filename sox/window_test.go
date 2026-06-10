package sox

import (
	"math"
	"testing"
)

func TestApplyHannSymmetric(t *testing.T) {
	// Hann over n points, m=n-1: endpoints 0, center 1.
	w := make([]float64, 5) // n=5, m=4
	for i := range w {
		w[i] = 1
	}
	applyHann(w)
	want := []float64{0, 0.5, 1, 0.5, 0}
	for i := range w {
		if math.Abs(w[i]-want[i]) > 1e-12 {
			t.Errorf("hann[%d]=%g want %g", i, w[i], want[i])
		}
	}
}

func TestMakeWindowFullNormalization(t *testing.T) {
	// For end=0 the empirical factor is 1, so sum(window)*2/sum == 2, i.e.
	// the normalized window sums to 2.
	ws := newWindowState(8, WindowHann)
	makeWindow(ws, 0)
	var s float64
	for i := 0; i < ws.dftSize; i++ {
		s += ws.window[i]
	}
	if math.Abs(s-2) > 1e-9 {
		t.Errorf("normalized window sum = %g, want 2", s)
	}
}

func TestMakeWindowReturnsRawSum(t *testing.T) {
	ws := newWindowState(8, WindowRectangular)
	// Rectangular window of n=dftSize+1 ones; sum over first dftSize taps = 8.
	if got := makeWindow(ws, 0); math.Abs(got-8) > 1e-12 {
		t.Errorf("rectangular raw sum = %g, want 8", got)
	}
}
