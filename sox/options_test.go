package sox

import "testing"

func TestNormalizeAppliesDefaults(t *testing.T) {
	got := normalize(Options{})
	if got.DBRange != 120 {
		t.Errorf("DBRange = %d, want 120", got.DBRange)
	}
	if got.Perm != 1 {
		t.Errorf("Perm = %d, want 1", got.Perm)
	}
	if got.Quantisation != 249 {
		t.Errorf("Quantisation = %d, want 249", got.Quantisation)
	}
	if got.Window != WindowHann {
		t.Errorf("Window = %v, want WindowHann", got.Window)
	}
}

func TestValidateRejectsBadCombos(t *testing.T) {
	cases := map[string]Options{
		"both y sizes":    {YSize: 257, YSizeTotal: 550},
		"kaiser v1":       {Window: WindowKaiser},
		"dolph v1":        {Window: WindowDolph},
		"non-p2 ysize":    {YSize: 200}, // dft = 398, not power of two
		"three time opts": {XSize: 800, PixelsPerSec: 100, Duration: 3},
		"negative xsize":  {XSize: -1},
		"negative dur":    {Duration: -1},
		"negative pps":    {PixelsPerSec: -1},
		"negative ytotal": {YSizeTotal: -1},
	}
	for name, opt := range cases {
		if err := validate(normalize(opt)); err == nil {
			t.Errorf("%s: expected error, got nil", name)
		}
	}
}

func TestValidateAcceptsDefaults(t *testing.T) {
	if err := validate(normalize(Options{})); err != nil {
		t.Errorf("default options rejected: %v", err)
	}
}
