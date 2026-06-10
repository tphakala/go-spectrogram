package sox

import (
	"math"
	"testing"
)

func TestDeriveDFTSize(t *testing.T) {
	// Default (YSizeTotal 0 => 550): y = max(32, 550-2)=548 -> dft 1024, rows 513.
	if dft, rows := deriveDFTSize(normalize(Options{})); dft != 1024 || rows != 513 {
		t.Errorf("default: dft=%d rows=%d, want 1024/513", dft, rows)
	}
	// -y 257 => dft = 2*256 = 512, rows 257.
	if dft, rows := deriveDFTSize(normalize(Options{YSize: 257})); dft != 512 || rows != 257 {
		t.Errorf("YSize257: dft=%d rows=%d, want 512/257", dft, rows)
	}
	// -Y 130 => y = max(32, 130-2)=128 -> dft 256, rows 129.
	if dft, rows := deriveDFTSize(normalize(Options{YSizeTotal: 130})); dft != 256 || rows != 129 {
		t.Errorf("YSizeTotal130: dft=%d rows=%d, want 256/129", dft, rows)
	}
}

func TestResolveTimeAxis(t *testing.T) {
	// Default, 3s known duration: x=800 then pps = min(5000, 800/3)=266.66...
	xs, pps := resolveTimeAxis(normalize(Options{}), 3.0)
	if xs != 800 {
		t.Errorf("xSize=%d, want 800", xs)
	}
	if pps < 266 || pps > 267 {
		t.Errorf("pps=%g, want ~266.67", pps)
	}
	// PixelsPerSec given, no x: x = round(pps*duration).
	xs2, pps2 := resolveTimeAxis(normalize(Options{PixelsPerSec: 100}), 2.5)
	if xs2 != 250 || pps2 != 100 {
		t.Errorf("xs2=%d pps2=%g, want 250/100", xs2, pps2)
	}
}

func TestStepSizing(t *testing.T) {
	// Hann dft=1024 raw sum ~= 512; rate 44100, pps 266.67.
	ws := newWindowState(1024, WindowHann)
	actual := makeWindow(ws, 0)
	step, blocks, norm := stepSizing(actual, 1024, 44100, 266.67, false)
	if step <= 0 || blocks <= 0 {
		t.Fatalf("step=%d blocks=%d", step, blocks)
	}
	if math.Abs(norm-1.0/float64(blocks)) > 1e-12 {
		t.Errorf("norm=%g, want %g", norm, 1.0/float64(blocks))
	}
	// Sanity: effective pps = rate/step/blocks should be near requested.
	eff := 44100.0 / float64(step) / float64(blocks)
	if eff < 100 || eff > 500 {
		t.Errorf("effective pps %g implausible", eff)
	}
}
