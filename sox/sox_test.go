package sox

import (
	"image"
	"os"
	"path/filepath"
	"testing"
)

func TestRenderDimensionsAndOrientation(t *testing.T) {
	const rate = 8000
	sig := tone(rate, 1.0, 500) // low tone -> energy near bottom rows
	img, err := Render(sig, rate, Options{Raw: true})
	if err != nil {
		t.Fatal(err)
	}
	_, rows := deriveDFTSize(normalize(Options{}))
	if img.Bounds().Dy() != rows {
		t.Errorf("height %d, want rows %d", img.Bounds().Dy(), rows)
	}
	if img.Bounds().Dx() == 0 {
		t.Fatal("zero width")
	}
	// 500 Hz is a low frequency: brightest pixel in column 0 should be near the
	// BOTTOM of the image (large y), confirming orientation flip.
	col := 0
	var brightestY, brightestIdx int
	for y := 0; y < rows; y++ {
		idx := int(img.ColorIndexAt(col, y))
		if idx > brightestIdx {
			brightestIdx, brightestY = idx, y
		}
	}
	if brightestY < rows/2 {
		t.Errorf("brightest row y=%d is in top half; orientation likely unflipped", brightestY)
	}
}

func TestRenderChrome(t *testing.T) {
	const rate = 8000
	sig := tone(rate, 1.0, 500)
	raw, err := Render(sig, rate, Options{Raw: true})
	if err != nil {
		t.Fatal(err)
	}
	img, err := Render(sig, rate, Options{})
	if err != nil {
		t.Fatal(err)
	}
	wantW, wantH := chromeDims(raw.Bounds().Dx(), raw.Bounds().Dy(), "")
	if img.Bounds().Dx() != wantW || img.Bounds().Dy() != wantH {
		t.Fatalf("chrome dims %dx%d, want %dx%d",
			img.Bounds().Dx(), img.Bounds().Dy(), wantW, wantH)
	}

	// Grid border: one pixel left of the raster, full raster height.
	// Bottom-up y in [below, below+rows) maps to image y in
	// [H-below-rows, H-below).
	H := img.Bounds().Dy()
	rows := raw.Bounds().Dy()
	for y := H - below - rows; y < H-below; y++ {
		if got := img.ColorIndexAt(left-1, y); got != gridIndex {
			t.Fatalf("grid pixel (%d,%d) = %d, want %d", left-1, y, got, gridIndex)
		}
	}

	// Legend bar: rightmost bar column holds spectrum colours (>= fixedPalette).
	barX := img.Bounds().Dx() - right - 1
	k := rows
	if k > 400 {
		k = 400
	}
	zbase := below + (rows-k)/2
	for yb := zbase; yb < zbase+k; yb++ {
		if got := img.ColorIndexAt(barX, H-1-yb); got < fixedPalette {
			t.Fatalf("legend pixel (%d,%d) = %d, want >= %d", barX, H-1-yb, got, fixedPalette)
		}
	}

	// Comment "Created by SoX" puts Text pixels in the bottom fontY rows.
	found := false
	for y := H - fontY; y < H && !found; y++ {
		for x := 0; x < img.Bounds().Dx() && !found; x++ {
			found = img.ColorIndexAt(x, y) == textIndex
		}
	}
	if !found {
		t.Error("no Text pixels in comment area")
	}

	// Title adds 20 rows.
	timg, err := Render(sig, rate, Options{Title: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	if timg.Bounds().Dy() != wantH+20 {
		t.Errorf("title height %d, want %d", timg.Bounds().Dy(), wantH+20)
	}
}

func TestRenderRejectsKaiser(t *testing.T) {
	if _, err := Render(tone(8000, 0.2, 500), 8000, Options{Window: WindowKaiser}); err == nil {
		t.Error("expected error for Kaiser window")
	}
}

func TestWritePNGRoundTrips(t *testing.T) {
	out := filepath.Join(t.TempDir(), "s.png")
	if err := WritePNG(out, tone(8000, 0.5, 1000), 8000, Options{}); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(out)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	cfg, format, err := image.DecodeConfig(f)
	if err != nil {
		t.Fatal(err)
	}
	if format != "png" {
		t.Errorf("format %q, want png", format)
	}
	if cfg.Width == 0 || cfg.Height == 0 {
		t.Errorf("empty image %dx%d", cfg.Width, cfg.Height)
	}
}
