package sox

import (
	"bytes"
	"image/png"
	"math/rand"
	"os"
	"os/exec"
	"testing"
)

// decodePalettedIndices loads a PNG and returns (width, height, indices) where
// indices[y*w+x] is the palette index. The PNG is 8-bit palette (color type 3).
func decodePalettedIndices(t *testing.T, path string) (int, int, []byte) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	img, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	idx := make([]byte, w*h)
	type ci interface{ ColorIndexAt(x, y int) uint8 }
	p, ok := img.(ci)
	if !ok {
		t.Fatalf("decoded image is not paletted")
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			idx[y*w+x] = p.ColorIndexAt(b.Min.X+x, b.Min.Y+y)
		}
	}
	return w, h, idx
}

// comparePNGs renders with both this package and the sox binary and compares
// every palette index. maxDelta is the largest per-pixel index difference the
// case tolerates; it is 0 (bit-exact) for everything except the documented
// exception in TestRasterParity.
func comparePNGsTol(t *testing.T, name string, samples []float32, rate int, opt Options, soxArgs []string, maxDelta int) {
	t.Helper()
	refPath := soxRunSpectrogram(t, samples, rate, true, soxArgs)
	opt.Raw = true // this harness compares against `sox ... spectrogram -r`
	rw, rh, ref := decodePalettedIndices(t, refPath)

	img, err := Render(samples, float64(rate), opt)
	if err != nil {
		t.Fatal(err)
	}
	gw, gh := img.Bounds().Dx(), img.Bounds().Dy()
	if gw != rw || gh != rh {
		t.Fatalf("%s: dims %dx%d != sox %dx%d", name, gw, gh, rw, rh)
	}
	var mismatch, worst int
	for y := 0; y < gh; y++ {
		for x := 0; x < gw; x++ {
			g := int(img.ColorIndexAt(x, y))
			r := int(ref[y*rw+x])
			d := g - r
			if d < 0 {
				d = -d
			}
			if d != 0 {
				mismatch++
				if d > worst {
					worst = d
				}
			}
		}
	}
	total := gw * gh
	matchPct := 100 * float64(total-mismatch) / float64(total)
	t.Logf("%s: %.3f%% exact, worst delta %d (%d/%d mismatched)", name, matchPct, worst, mismatch, total)
	// The renderer is bit-exact against the binary, so assert exactly that.
	// A looser bound would let the README's "100.000% of palette indices"
	// claim rot silently: the float32 FFT this replaced sat at 99.910%, which
	// the old >=99.5% threshold accepted without complaint.
	if maxDelta == 0 && mismatch != 0 {
		t.Errorf("%s: %.3f%% exact, %d/%d pixels differ, worst delta %d (want bit-exact)",
			name, matchPct, mismatch, total, worst)
	}
	if worst > maxDelta {
		t.Errorf("%s: worst palette-index delta %d (want <=%d)", name, worst, maxDelta)
	}
	// A delta allowance on its own would accept every pixel drifting by one, so
	// the tolerated cases also have to hold their match rate. The documented
	// exception sits at 99.920%.
	if maxDelta > 0 && matchPct < 99.5 {
		t.Errorf("%s: only %.3f%% exact (want >=99.5%% even with a delta allowance)", name, matchPct)
	}
}

func sweep(rate int, dur float64, f0, f1 float64) []float32 {
	n := int(float64(rate) * dur)
	s := make([]float32, n)
	for i := range s {
		tt := float64(i) / float64(rate)
		f := f0 + (f1-f0)*tt/dur
		s[i] = float32(0.5 * mathSin(2*pi*f*tt))
	}
	return s
}

func noise(rate int, dur float64, seed int64) []float32 {
	n := int(float64(rate) * dur)
	s := make([]float32, n)
	rng := rand.New(rand.NewSource(seed))
	for i := range s {
		s[i] = float32((rng.Float64()*2 - 1) * 0.3)
	}
	return s
}

// soxSpectrogramSupportsNormalize probes whether the installed sox binary
// recognises the spectrogram -n (normalize) flag. It was added after v14.4.2;
// the flag is absent from the installed 14.4.2 binary but present in 14.4.3+.
// We probe via --help-effect which lists all spectrogram flags.
func soxSpectrogramSupportsNormalize() bool {
	cmd := exec.Command("sox", "--help-effect", "spectrogram")
	out, _ := cmd.CombinedOutput()
	// The normalize flag appears in help as "\t-n\t" (tab-n-tab).
	return bytes.Contains(out, []byte("\t-n\t"))
}

func comparePNGs(t *testing.T, name string, samples []float32, rate int, opt Options, soxArgs []string) {
	t.Helper()
	comparePNGsTol(t, name, samples, rate, opt, soxArgs, 0)
}

func TestRasterParity(t *testing.T) {
	requireSox(t)
	const rate = 16000
	cases := []struct {
		name     string
		samples  []float32
		opt      Options
		args     []string
		maxDelta int // 0 = bit-exact; see zrange-max for the one exception
	}{
		{"default-sweep", sweep(rate, 2.0, 200, 6000), Options{}, nil, 0},
		{"normalize", sweep(rate, 2.0, 200, 6000), Options{Normalize: true}, []string{"-n"}, 0},
		{"xtrunc", sweep(rate, 3.0, 200, 6000), Options{XSize: 120}, []string{"-x", "120"}, 0},
		{"ysize512", noise(rate, 1.5, 1), Options{YSize: 257}, []string{"-y", "257"}, 0},
		{"zrange90", sweep(rate, 2.0, 200, 6000), Options{DBRange: 90}, []string{"-z", "90"}, 0},
		{"gain20", sweep(rate, 2.0, 200, 6000), Options{Gain: 20}, []string{"-Z", "20"}, 0},
		{"mono-pal", sweep(rate, 2.0, 200, 6000), Options{Monochrome: true}, []string{"-m"}, 0},
		{"highcolour", sweep(rate, 2.0, 200, 6000), Options{HighColour: true}, []string{"-h"}, 0},
		// SlackOverlap is the only geometry-affecting field; exercise it end-to-end.
		{"slack", sweep(rate, 2.0, 200, 6000), Options{SlackOverlap: true}, []string{"-s"}, 0},
		// Two lengths to exercise both branches of the drain remainder logic.
		{"drain-a", noise(rate, 1.37, 7), Options{}, nil, 0},
		{"drain-b", noise(rate, 1.61, 9), Options{}, nil, 0},
		// The dynamic-range extremes validate() permits. The upper end matters
		// most: at -z 180 a bin 150 dB down is still a distinguishable colour,
		// so it is the case where a transform that flushed low-energy bins
		// would actually show.
		{"zrange-min", sweep(rate, 2.0, 200, 6000), Options{DBRange: 20}, []string{"-z", "20"}, 0},
		// PRE-EXISTING, not caused by this package's transform: at the maximum
		// dynamic range validate() allows, 329/410400 pixels land one palette
		// index off SoX. The count is byte-identical on the float32 FFT this
		// replaced, so the cause is palette quantisation at the extreme range,
		// not the DFT. Tracked separately; asserted at delta <= 1 so a real
		// regression here still fails.
		{"zrange-max", sweep(rate, 2.0, 200, 6000), Options{DBRange: 180}, []string{"-z", "180"}, 1},
		// A width that is not a multiple of the raster transpose's 64-wide
		// tile, so the ragged edge is checked pixel-exact against the oracle
		// rather than merely for absence of a panic.
		{"xsize-ragged", sweep(rate, 2.0, 200, 6000), Options{XSize: 517}, []string{"-x", "517"}, 0},
	}
	normalizeSupported := soxSpectrogramSupportsNormalize()
	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			if c.name == "normalize" && !normalizeSupported {
				t.Skip("installed sox does not support spectrogram -n (normalize); added in 14.4.3+")
			}
			comparePNGsTol(t, c.name, c.samples, rate, c.opt, c.args, c.maxDelta)
		})
	}
}

// compareFullPNGs renders the full chrome PNG and compares against sox without
// -r. Chrome pixels (everything outside the raster rectangle) are deterministic
// and must match exactly; the raster keeps the v1 thresholds.
func compareFullPNGs(t *testing.T, name string, samples []float32, rate int, opt Options, soxArgs []string) {
	t.Helper()
	refPath := soxRunSpectrogram(t, samples, rate, false, soxArgs)
	rw, rh, ref := decodePalettedIndices(t, refPath)

	img, err := Render(samples, float64(rate), opt)
	if err != nil {
		t.Fatal(err)
	}
	gw, gh := img.Bounds().Dx(), img.Bounds().Dy()
	if gw != rw || gh != rh {
		t.Fatalf("%s: dims %dx%d != sox %dx%d", name, gw, gh, rw, rh)
	}

	// Reconstruct the raster rectangle in top-down image coordinates.
	rasterCols := gw - (left + between + spectrumWidth + right)
	rasterRows := gh - below - 30
	if opt.Title != "" {
		rasterRows -= 20
	}
	x0, x1 := left, left+rasterCols
	y0, y1 := gh-below-rasterRows, gh-below

	var mismatch, worst, chromeBad int
	for y := 0; y < gh; y++ {
		for x := 0; x < gw; x++ {
			g := int(img.ColorIndexAt(x, y))
			r := int(ref[y*rw+x])
			d := g - r
			if d < 0 {
				d = -d
			}
			if d == 0 {
				continue
			}
			if x >= x0 && x < x1 && y >= y0 && y < y1 {
				mismatch++
				if d > worst {
					worst = d
				}
			} else {
				chromeBad++
				if chromeBad <= 5 {
					t.Errorf("%s: chrome pixel (%d,%d) = %d, sox %d", name, x, y, g, r)
				}
			}
		}
	}
	// Chrome is drawn from the same font and tick math as SoX, so it matches
	// exactly; asserting that keeps the README's "chrome and raster alike"
	// claim honest.
	if chromeBad != 0 {
		t.Errorf("%s: %d chrome pixels differ (want 0)", name, chromeBad)
	}
	total := rasterCols * rasterRows
	matchPct := 100 * float64(total-mismatch) / float64(total)
	t.Logf("%s: raster %.3f%% exact, worst delta %d; chrome mismatches %d",
		name, matchPct, worst, chromeBad)
	if mismatch != 0 {
		t.Errorf("%s: raster %.3f%% exact, %d/%d pixels differ, worst delta %d (want bit-exact)",
			name, matchPct, mismatch, total, worst)
	}
}

func TestFullParity(t *testing.T) {
	requireSox(t)
	const rate = 16000
	cases := []struct {
		name    string
		samples []float32
		rateHz  int
		opt     Options
		args    []string
	}{
		{"full-default", sweep(rate, 2.0, 200, 6000), rate, Options{}, nil},
		{"title-comment", sweep(rate, 2.0, 200, 6000), rate,
			Options{Title: "Sweep test", Comment: "hello world"},
			[]string{"-t", "Sweep test", "-c", "hello world"}},
		{"no-axes", sweep(rate, 2.0, 200, 6000), rate, Options{NoAxes: true}, []string{"-a"}},
		{"mono", sweep(rate, 2.0, 200, 6000), rate, Options{Monochrome: true}, []string{"-m"}},
		{"light", sweep(rate, 2.0, 200, 6000), rate, Options{LightBackground: true}, []string{"-l"}},
		{"high-colour", sweep(rate, 2.0, 200, 6000), rate, Options{HighColour: true}, []string{"-h"}},
		{"perm3-high", sweep(rate, 2.0, 200, 6000), rate,
			Options{HighColour: true, Perm: 3}, []string{"-h", "-p", "3"}},
		{"zrange80", sweep(rate, 2.0, 200, 6000), rate, Options{DBRange: 80}, []string{"-z", "80"}},
		{"gain10", sweep(rate, 2.0, 200, 6000), rate, Options{Gain: 10}, []string{"-Z", "10"}},
		// rate 1600 -> Nyquist 800 Hz -> plain "Hz" axis (no k prefix).
		{"hz-prefix", sweep(1600, 2.0, 50, 700), 1600, Options{}, nil},
		{"ysize", noise(rate, 1.5, 3), rate, Options{YSize: 257}, []string{"-y", "257"}},
		{"xtrunc", sweep(rate, 3.0, 200, 6000), rate, Options{XSize: 120}, []string{"-x", "120"}},
	}
	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			compareFullPNGs(t, c.name, c.samples, c.rateHz, c.opt, c.args)
		})
	}
}
