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
		"both y sizes":     {YSize: 257, YSizeTotal: 550},
		"kaiser v1":        {Window: WindowKaiser},
		"dolph v1":         {Window: WindowDolph},
		"non-p2 ysize":     {YSize: 200}, // dft = 398, not power of two
		"ysize below 64":   {YSize: 17},  // pow2 dft but under SoX's -y minimum
		"ysize too large":  {YSize: 200001},
		"xsize below 100":  {XSize: 99}, // SoX -x minimum
		"xsize too large":  {XSize: 200001},
		"pps below 1":      {PixelsPerSec: 0.5}, // SoX -X minimum
		"pps too large":    {PixelsPerSec: 5001},
		"ytotal below 130": {YSizeTotal: 129}, // SoX -Y minimum
		"ytotal too large": {YSizeTotal: 200001},
		"three time opts":  {XSize: 800, PixelsPerSec: 100, Duration: 3},
		"negative xsize":   {XSize: -1},
		"negative dur":     {Duration: -1},
		"negative pps":     {PixelsPerSec: -1},
		"negative ytotal":  {YSizeTotal: -1},
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

func TestValidateYSizeBounds(t *testing.T) {
	for _, y := range []int{65, 257} {
		if err := validate(normalize(Options{YSize: y})); err != nil {
			t.Errorf("YSize %d rejected: %v", y, err)
		}
	}
}

// Regression: YSize values below SoX's -y minimum of 64 used to pass
// validation and panic inside drawChrome (the rotated frequency-axis label
// needs more raster rows than a tiny YSize provides).
func TestRenderRejectsSmallYSize(t *testing.T) {
	if _, err := Render(tone(8000, 0.2, 500), 8000, Options{YSize: 17}); err == nil {
		t.Error("expected error for YSize 17, got nil")
	}
}

func TestNormalizeCommentDefault(t *testing.T) {
	if got := normalize(Options{}).Comment; got != "Created by SoX" {
		t.Errorf("default Comment = %q, want %q", got, "Created by SoX")
	}
	if got := normalize(Options{Comment: "hi"}).Comment; got != "hi" {
		t.Errorf("Comment = %q, want %q", got, "hi")
	}
}
