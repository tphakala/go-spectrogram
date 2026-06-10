package sox

import "testing"

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
