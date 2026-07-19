package sox

import (
	"math"
	"testing"
)

// A pure tone should put almost all energy in one frequency row; that row's
// dBFS should be much higher than a far-away row.
func TestAnalyzerToneConcentratesEnergy(t *testing.T) {
	const rate = 8000.0
	const dur = 1.0
	n := int(rate * dur)
	sig := make([]float32, n)
	for i := range sig {
		sig[i] = float32(0.5 * math.Sin(2*math.Pi*1000*float64(i)/rate))
	}
	opt := normalize(Options{})
	dft, rows := deriveDFTSize(opt)
	xSize, pps := resolveTimeAxis(opt, dur)
	ws := newWindowState(dft, opt.Window)
	actual := makeWindow(ws, 0)
	step, blocks, norm := stepSizing(actual, dft, rate, pps, opt.SlackOverlap)

	a, err := newAnalyzer(dft, rows, step, blocks, norm, -opt.Gain, opt.DBRange, ws, xSize)
	if err != nil {
		t.Fatal(err)
	}
	cols := a.run(sig)
	if cols == 0 {
		t.Fatal("no columns produced")
	}
	// Frequency bin for 1000 Hz: i = f / (rate/dft).
	binHz := rate / float64(dft)
	toneRow := int(1000/binHz + 0.5)
	farRow := toneRow + rows/4
	// Compare column 0.
	tone := a.dBfs[0*rows+toneRow]
	far := a.dBfs[0*rows+farRow]
	if !(tone > far+30) {
		t.Errorf("tone row %d (%.1f dB) not >> far row %d (%.1f dB)", toneRow, tone, farRow, far)
	}
}

func TestAnalyzerColumnCountMatchesGeometry(t *testing.T) {
	const rate = 8000.0
	const dur = 2.0
	n := int(rate * dur)
	sig := make([]float32, n) // silence is fine for counting
	opt := normalize(Options{})
	dft, rows := deriveDFTSize(opt)
	xSize, pps := resolveTimeAxis(opt, dur)
	ws := newWindowState(dft, opt.Window)
	actual := makeWindow(ws, 0)
	step, blocks, norm := stepSizing(actual, dft, rate, pps, opt.SlackOverlap)
	a, err := newAnalyzer(dft, rows, step, blocks, norm, -opt.Gain, opt.DBRange, ws, xSize)
	if err != nil {
		t.Fatal(err)
	}
	cols := a.run(sig)
	// Expected columns ~ rate*dur / (step*blocks); never exceeds xSize.
	exp := int(rate * dur / float64(step*blocks))
	if cols > xSize {
		t.Errorf("cols %d exceeds xSize %d", cols, xSize)
	}
	if cols < exp-2 || cols > exp+2 {
		t.Errorf("cols %d not within 2 of expected %d", cols, exp)
	}
}

// TestNewAnalyzerRejectsBinMismatch guards the invariant that rows equals the
// DFT bin count. rows sizes every per-column buffer, and the simd reductions in
// doColumn process only min(len(dst), len(src)) elements, so a mismatch would
// quietly truncate each column rather than fail.
func TestNewAnalyzerRejectsBinMismatch(t *testing.T) {
	ws := newWindowState(1024, WindowHann)
	makeWindow(ws, 0)
	if _, err := newAnalyzer(1024, 512, 256, 1, 1, 0, 120, ws, 100); err == nil {
		t.Fatal("expected an error when rows != dftSize/2+1, got nil")
	}
	if _, err := newAnalyzer(1024, 513, 256, 1, 1, 0, 120, ws, 100); err != nil {
		t.Fatalf("expected the correct bin count to be accepted, got %v", err)
	}
}
