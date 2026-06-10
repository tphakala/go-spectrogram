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

func comparePNGs(t *testing.T, name string, samples []float32, rate int, opt Options, soxArgs []string) {
	t.Helper()
	refPath := soxRunSpectrogram(t, samples, rate, soxArgs)
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
	if matchPct < 99.5 {
		t.Errorf("%s: only %.3f%% exact (want >=99.5%%)", name, matchPct)
	}
	if worst > 1 {
		t.Errorf("%s: worst palette-index delta %d (want <=1)", name, worst)
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

func TestRasterParity(t *testing.T) {
	requireSox(t)
	const rate = 16000
	cases := []struct {
		name    string
		samples []float32
		opt     Options
		args    []string
	}{
		{"default-sweep", sweep(rate, 2.0, 200, 6000), Options{}, nil},
		{"normalize", sweep(rate, 2.0, 200, 6000), Options{Normalize: true}, []string{"-n"}},
		{"xtrunc", sweep(rate, 3.0, 200, 6000), Options{XSize: 120}, []string{"-x", "120"}},
		{"ysize512", noise(rate, 1.5, 1), Options{YSize: 257}, []string{"-y", "257"}},
		{"zrange90", sweep(rate, 2.0, 200, 6000), Options{DBRange: 90}, []string{"-z", "90"}},
		{"gain20", sweep(rate, 2.0, 200, 6000), Options{Gain: 20}, []string{"-Z", "20"}},
		{"mono-pal", sweep(rate, 2.0, 200, 6000), Options{Monochrome: true}, []string{"-m"}},
		// Two lengths to exercise both branches of the drain remainder logic.
		{"drain-a", noise(rate, 1.37, 7), Options{}, nil},
		{"drain-b", noise(rate, 1.61, 9), Options{}, nil},
	}
	normalizeSupported := soxSpectrogramSupportsNormalize()
	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			if c.name == "normalize" && !normalizeSupported {
				t.Skip("installed sox does not support spectrogram -n (normalize); added in 14.4.3+")
			}
			comparePNGs(t, c.name, c.samples, rate, c.opt, c.args)
		})
	}
}
