package sox

import (
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// paletteHistogram counts how many pixels carry each palette index.
func paletteHistogram(img *image.Paletted) map[uint8]int {
	h := make(map[uint8]int)
	for _, p := range img.Pix {
		h[p]++
	}
	return h
}

// checkInPalette is the liveness floor: no pixel may index outside the palette.
func checkInPalette(t *testing.T, img *image.Paletted) {
	t.Helper()
	for i, p := range img.Pix {
		if int(p) >= len(img.Palette) {
			t.Fatalf("pixel %d index %d out of palette range %d", i, p, len(img.Palette))
		}
	}
}

// Edge cases the SoX parity sweep cannot reach, because they are inputs the
// binary cannot be handed identically (non-finite samples) or degenerate
// lengths. Parity now covers the dynamic-range extremes and the ragged tile
// width pixel-exactly, so this file only carries what parity cannot.
//
// These assert image CONTENT, not merely absence of a panic. An earlier version
// checked only that indices were within the palette, which fires on 1 of 256
// byte values and let both a halved-palette mutation and a vertically mirrored
// raster pass unnoticed.
func TestRenderEdgeCases(t *testing.T) {
	const rate = 24000
	cases := []struct {
		name    string
		samples []float32
		opt     Options
		// wantUniform is true when a correct render is essentially one flat
		// colour, because digital silence has no signal anywhere.
		wantUniform bool
	}{
		{"shorter-than-one-dft", make([]float32, 100), Options{XSize: 258, YSize: 129}, true},
		{"exactly-one-sample", make([]float32, 1), Options{XSize: 258, YSize: 129}, true},
		{"all-silence-15s", make([]float32, 15*rate), Options{XSize: 1026, YSize: 513}, true},
		{"silence-normalized", make([]float32, rate), Options{XSize: 258, YSize: 129, Normalize: true}, true},
		{"tone-dbrange-min", benchSignal(rate, rate), Options{XSize: 258, YSize: 129, DBRange: 20}, false},
		{"tone-dbrange-max", benchSignal(rate, rate), Options{XSize: 258, YSize: 129, DBRange: 180}, false},
		{"tone-normalized", benchSignal(rate, rate), Options{XSize: 258, YSize: 129, Normalize: true}, false},
		{"tone-raw", benchSignal(rate, rate), Options{XSize: 514, YSize: 257, Raw: true}, false},
		{"tone-ragged-width", benchSignal(rate, rate), Options{XSize: 517, YSize: 129}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			img, err := Render(c.samples, rate, c.opt)
			if err != nil {
				t.Fatalf("Render: %v", err)
			}
			checkInPalette(t, img)

			b := img.Bounds()
			switch {
			case c.opt.Raw:
				// Raw output is exactly the raster: one row per DFT bin.
				// Asserting the height exactly catches a geometry regression
				// that a "> 0" check would not.
				if b.Dy() != c.opt.YSize {
					t.Errorf("raw height %d, want YSize %d", b.Dy(), c.opt.YSize)
				}
				if b.Dx() <= 0 || b.Dx() > c.opt.XSize {
					t.Errorf("raw width %d, want 1..%d", b.Dx(), c.opt.XSize)
				}
			default:
				// XSize is an upper bound on the raster width, not a fixed
				// width: a clip shorter than the DFT yields fewer columns and
				// a correspondingly narrower image. The height is fixed
				// though, so chrome must always add rows above YSize.
				if b.Dy() <= c.opt.YSize {
					t.Errorf("height %d does not enclose a %d-row raster plus chrome",
						b.Dy(), c.opt.YSize)
				}
				if b.Dx() <= 0 {
					t.Errorf("width %d, want > 0", b.Dx())
				}
			}

			hist := paletteHistogram(img)
			if c.wantUniform {
				// Silence renders as the floor colour everywhere. With chrome
				// the axes contribute their own indices, so require one
				// dominant index rather than literally one.
				top := 0
				for _, n := range hist {
					if n > top {
						top = n
					}
				}
				if frac := float64(top) / float64(len(img.Pix)); frac < 0.5 {
					t.Errorf("silence: most common palette index covers only %.1f%% of the image, want a flat render", frac*100)
				}
			} else if len(hist) < 3 {
				// A signal-bearing render must show structure. This is what
				// catches a raster collapsed to a single colour, which the old
				// palette-range assertion accepted without complaint.
				t.Errorf("signal render uses only %d palette indices, want a non-uniform raster", len(hist))
			}
		})
	}
}

// TestRenderToneLandsAtExpectedRow pins the frequency-axis orientation. SoX puts
// DC at the bottom row and Nyquist at the top, so a single tone must brighten
// one specific row. A vertically mirrored raster still produces a plausible
// image with a plausible palette histogram and passes every other test here;
// this is the assertion that fails it.
func TestRenderToneLandsAtExpectedRow(t *testing.T) {
	const (
		rate = 24000
		tone = 3000.0
		ys   = 129
	)
	sig := make([]float32, 2*rate)
	for i := range sig {
		sig[i] = float32(0.5 * math.Sin(2*math.Pi*tone*float64(i)/rate))
	}
	img, err := Render(sig, rate, Options{XSize: 258, YSize: ys, Raw: true})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	b := img.Bounds()
	if b.Dy() != ys {
		t.Fatalf("height %d, want %d", b.Dy(), ys)
	}

	// Find the brightest row in a column near the middle of the image.
	x := b.Min.X + b.Dx()/2
	bright, brightY := -1, -1
	for y := b.Min.Y; y < b.Max.Y; y++ {
		if v := int(img.ColorIndexAt(x, y)); v > bright {
			bright, brightY = v, y
		}
	}

	// Raster row 0 is DC and the image is stored top-down with DC at the
	// bottom, so the tone's bin counts up from the bottom edge.
	binsPerHz := float64(ys-1) / (rate / 2.0)
	wantFromBottom := int(math.Round(tone * binsPerHz))
	wantY := b.Max.Y - 1 - wantFromBottom
	if d := brightY - wantY; d < -2 || d > 2 {
		t.Errorf("tone at %.0f Hz brightened row y=%d, want y=%d (+/-2); "+
			"a mismatch this large means the frequency axis is flipped or shifted",
			tone, brightY, wantY)
	}
}

func TestRenderNaNInf(t *testing.T) {
	const rate = 24000
	for _, c := range []struct {
		name string
		v    float32
	}{
		{"nan", float32(math.NaN())},
		{"posinf", float32(math.Inf(1))},
		{"neginf", float32(math.Inf(-1))},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := benchSignal(rate, rate)
			for i := 0; i < len(s); i += 977 {
				s[i] = c.v
			}
			img, err := Render(s, rate, Options{XSize: 258, YSize: 129})
			if err != nil {
				t.Fatalf("Render: %v", err)
			}
			// Non-finite input must not escape as an out-of-palette index; the
			// dB path maps NaN to the floor colour rather than int(NaN).
			checkInPalette(t, img)
		})
	}
}

// TestRenderNormalizeWithNonFinite combines two options the table above
// exercises only separately: with Normalize on, a non-finite sample can reach
// the auto-gain reference itself.
func TestRenderNormalizeWithNonFinite(t *testing.T) {
	const rate = 24000
	s := benchSignal(rate, rate)
	for i := 0; i < len(s); i += 501 {
		s[i] = float32(math.Inf(1))
	}
	img, err := Render(s, rate, Options{XSize: 258, YSize: 129, Normalize: true})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	checkInPalette(t, img)
}

// TestWritePNGIsAtomic covers the publish contract: the destination either does
// not exist or holds a complete PNG, and no temporary file is left behind.
func TestWritePNGIsAtomic(t *testing.T) {
	const rate = 24000
	dir := t.TempDir()
	out := filepath.Join(dir, "spec.png")

	if err := WritePNG(out, benchSignal(rate, rate), rate, Options{XSize: 258, YSize: 129}); err != nil {
		t.Fatalf("WritePNG: %v", err)
	}

	// The written file must decode as a complete image, not a truncated one.
	f, err := os.Open(out)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer func() {
		if err := f.Close(); err != nil {
			t.Errorf("close: %v", err)
		}
	}()
	if _, err := png.Decode(f); err != nil {
		t.Fatalf("decode written PNG: %v", err)
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != "spec.png" {
		var names []string
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Errorf("directory holds %v, want only spec.png (a leftover temp file means the rename path is wrong)", names)
	}
}

// TestWritePNGFailureLeavesNoFile covers the error path: a render that cannot be
// encoded must not publish anything at the destination.
func TestWritePNGFailureLeavesNoFile(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "spec.png")

	// Zero columns yields a zero-width image, which the PNG encoder rejects.
	err := WritePNG(out, nil, 24000, Options{XSize: 258, YSize: 129, Raw: true})
	if err == nil {
		t.Fatal("expected an encode error for a zero-width image, got nil")
	}
	if _, statErr := os.Stat(out); !os.IsNotExist(statErr) {
		t.Errorf("destination exists after a failed write (stat err %v), want no file", statErr)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Errorf("temp files left behind: %d entries", len(entries))
	}
}

// TestWritePNGFileMode pins the permissions of the published file. Writing via
// os.CreateTemp and renaming is the correct way to publish atomically, but
// CreateTemp makes the file 0600 where the os.Create it replaced produced
// 0666&^umask; a consumer serving these files as another user would break.
func TestWritePNGFileMode(t *testing.T) {
	const rate = 24000
	out := filepath.Join(t.TempDir(), "spec.png")
	if err := WritePNG(out, benchSignal(rate, rate), rate, Options{XSize: 258, YSize: 129}); err != nil {
		t.Fatalf("WritePNG: %v", err)
	}
	fi, err := os.Stat(out)
	if err != nil {
		t.Fatal(err)
	}
	if got := fi.Mode().Perm(); got&0o044 == 0 {
		t.Errorf("mode %04o is not group/world readable; a consumer running as another user cannot serve it", got)
	}
}
