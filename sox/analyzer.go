package sox

import (
	"math"

	"github.com/tphakala/go-spectrogram/internal/dsp"
)

// analyzer reproduces SoX's streaming spectrogram accumulation. dBfs is stored
// column-major: dBfs[col*rows + row], row 0 = DC. max tracks the brightest
// pre-gain dBFS (for -n normalize) and starts at -dBRange.
type analyzer struct {
	dftSize, rows        int
	stepSize, blockSteps int
	blockNorm            float64
	gain                 int // SoX p->gain = -opt.Gain
	dBRange              int
	ws                   *windowState
	xSize                int

	plan *dsp.FFTPlan
	buf  []float64   // len dftSize
	cin  []complex64 // FFT input scratch, len dftSize
	mag  []float64   // len rows

	read      int
	end       int
	endMin    int
	lastEnd   int
	blockNum  int
	cols      int
	truncated bool

	dBfs []float32
	max  float64
}

func newAnalyzer(dftSize, rows, stepSize, blockSteps int, blockNorm float64, gain, dBRange int, ws *windowState, xSize int) *analyzer {
	a := &analyzer{
		dftSize: dftSize, rows: rows,
		stepSize: stepSize, blockSteps: blockSteps, blockNorm: blockNorm,
		gain: gain, dBRange: dBRange, ws: ws, xSize: xSize,
		plan: dsp.NewFFTPlan(dftSize),
		buf:  make([]float64, dftSize),
		cin:  make([]complex64, dftSize),
		mag:  make([]float64, rows),
	}
	a.end = dftSize                   // spectrogram.c:429
	a.endMin = 0                      // zeroed in start
	a.lastEnd = 0                     // make_window(p, 0) already done before loop
	a.max = -float64(dBRange)         // spectrogram.c:443
	a.read = (stepSize - dftSize) / 2 // spectrogram.c:444 (negative)
	return a
}

// run feeds all samples then drains, returning the number of columns produced.
func (a *analyzer) run(samples []float32) int {
	a.flow(samples)
	a.drain()
	return a.cols
}

// flow ports spectrogram.c:497-531. `in` may be float32 (audio) fed as float64.
func (a *analyzer) flow(in []float32) {
	idx, n := 0, len(in)
	for !a.truncated {
		if a.read == a.stepSize {
			copy(a.buf[:a.dftSize-a.stepSize], a.buf[a.stepSize:a.dftSize])
			a.read = 0
		}
		for idx < n && a.read < a.stepSize {
			a.buf[a.dftSize-a.stepSize+a.read] = float64(in[idx])
			idx++
			a.read++
			a.end--
		}
		if a.read != a.stepSize {
			break
		}
		a.processBlock()
	}
}

// processBlock windows, FFTs, accumulates magnitudes, and emits a column every
// blockSteps DFTs.
func (a *analyzer) processBlock() {
	if a.end < a.endMin {
		a.end = a.endMin
	}
	if a.end != a.lastEnd {
		makeWindow(a.ws, a.end)
		a.lastEnd = a.end
	}
	for i := 0; i < a.dftSize; i++ {
		a.cin[i] = complex(float32(a.buf[i]*a.ws.window[i]), 0)
	}
	spec := a.plan.Forward(a.cin)
	half := a.dftSize >> 1
	a.mag[0] += sq(float64(real(spec[0])))
	for i := 1; i < half; i++ {
		a.mag[i] += sq(float64(real(spec[i]))) + sq(float64(imag(spec[i])))
	}
	a.mag[half] += sq(float64(real(spec[half])))

	a.blockNum++
	if a.blockNum == a.blockSteps {
		a.doColumn()
	}
}

// doColumn ports spectrogram.c:449-475 (non-truncate variant).
func (a *analyzer) doColumn() {
	if a.cols == a.xSize {
		a.truncated = true
		return
	}
	a.cols++
	for i := 0; i < a.rows; i++ {
		dBfs := 10 * math.Log10(a.mag[i]*a.blockNorm)
		a.dBfs = append(a.dBfs, float32(dBfs+float64(a.gain)))
		if dBfs > a.max {
			a.max = dBfs
		}
	}
	for i := range a.mag {
		a.mag[i] = 0
	}
	a.blockNum = 0
}

// drain ports spectrogram.c:536-566.
func (a *analyzer) drain() {
	if a.truncated {
		return
	}
	isamp := (a.dftSize - a.stepSize) / 2
	leftOver := (isamp + a.read) % a.stepSize
	if leftOver >= a.stepSize>>1 {
		isamp += a.stepSize - leftOver
	}
	a.end = 0
	a.endMin = -a.dftSize
	a.flow(make([]float32, isamp))
	if !a.truncated && a.blockNum != 0 {
		a.blockNorm *= float64(a.blockSteps) / float64(a.blockNum)
		a.doColumn()
	}
}

func sq(x float64) float64 { return x * x }
