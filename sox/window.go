package sox

import "math"

// windowState holds the reusable window buffer (length dftSize+1) and the
// window selection. makeWindow rewrites window[] for a given `end`.
type windowState struct {
	dftSize int
	winType WindowType
	window  []float64 // len dftSize+1
}

func newWindowState(dftSize int, w WindowType) *windowState {
	return &windowState{dftSize: dftSize, winType: w, window: make([]float64, dftSize+1)}
}

// applyHann multiplies h in place by a symmetric Hann window of length len(h)
// (effects_i_dsp.c:264). m = len(h)-1.
func applyHann(h []float64) {
	m := len(h) - 1
	for i := range h {
		h[i] *= 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(m))
	}
}

// applyHamming (effects_i_dsp.c:273).
func applyHamming(h []float64) {
	m := len(h) - 1
	for i := range h {
		h[i] *= 0.53836 - 0.46164*math.Cos(2*math.Pi*float64(i)/float64(m))
	}
}

// applyBartlett (effects_i_dsp.c:282).
func applyBartlett(h []float64) {
	m := float64(len(h) - 1)
	for i := range h {
		h[i] *= 2.0 / m * (m/2 - math.Abs(float64(i)-m/2))
	}
}

// makeWindow rewrites ws.window for the given `end` and returns the RAW window
// sum (before normalization). Ports spectrogram.c:280-317 in full, including
// the end != 0 path used for the centering ramp (positive end) and the drain
// flush (negative end).
func makeWindow(ws *windowState, end int) float64 {
	dft := ws.dftSize
	abs := end
	if abs < 0 {
		abs = -abs
	}
	n := 1 + dft - abs

	// w points at window[end] for end >= 0, else window[0]. (end<0 -> w=window.)
	off := 0
	if end >= 0 {
		off = end
	}
	if end != 0 {
		for i := range ws.window {
			ws.window[i] = 0
		}
	}
	for i := 0; i < n; i++ {
		ws.window[off+i] = 1
	}
	switch ws.winType {
	case WindowHann:
		applyHann(ws.window[off : off+n])
	case WindowHamming:
		applyHamming(ws.window[off : off+n])
	case WindowBartlett:
		applyBartlett(ws.window[off : off+n])
	case WindowRectangular:
		// no-op
	}

	var sum float64
	for i := 0; i < dft; i++ {
		sum += ws.window[i]
	}
	n-- // SoX's --n
	for i := 0; i < dft; i++ {
		f := float64(n) / float64(dft)
		ws.window[i] *= 2 / sum * f * f
	}
	return sum
}
