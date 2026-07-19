package sox

import (
	"math"
	"testing"
)

// Edge cases the parity sweep does not cover: degenerate lengths, silence,
// dynamic-range extremes, autogain, and non-finite samples. These assert the
// renderer stays in bounds and does not panic, not that it matches SoX.
func TestRenderEdgeCases(t *testing.T) {
	cases := []struct {
		name    string
		samples []float32
		opt     Options
	}{
		{"shorter-than-one-dft", make([]float32, 100), Options{XSize: 258, YSize: 129}},
		{"exactly-one-sample", make([]float32, 1), Options{XSize: 258, YSize: 129}},
		{"all-silence-15s", make([]float32, 15*24000), Options{XSize: 1026, YSize: 513}},
		{"dbrange-min", benchSignal(24000, 24000), Options{XSize: 258, YSize: 129, DBRange: 20}},
		{"dbrange-max", benchSignal(24000, 24000), Options{XSize: 258, YSize: 129, DBRange: 180}},
		{"normalize", benchSignal(24000, 24000), Options{XSize: 258, YSize: 129, Normalize: true}},
		{"normalize-silence", make([]float32, 24000), Options{XSize: 258, YSize: 129, Normalize: true}},
		{"raw-offsets", benchSignal(24000, 24000), Options{XSize: 514, YSize: 257, Raw: true}},
		{"non-multiple-of-tile", benchSignal(24000, 24000), Options{XSize: 517, YSize: 129}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			img, err := Render(c.samples, 24000, c.opt)
			if err != nil {
				t.Fatalf("Render: %v", err)
			}
			if img == nil {
				t.Fatal("nil image")
			}
			b := img.Bounds()
			if b.Dx() <= 0 || b.Dy() <= 0 {
				t.Fatalf("degenerate bounds %v", b)
			}
			for i, p := range img.Pix {
				if int(p) >= len(img.Palette) {
					t.Fatalf("pixel %d index %d out of palette range %d", i, p, len(img.Palette))
				}
			}
		})
	}
}

func TestRenderNaNInf(t *testing.T) {
	for _, c := range []struct {
		name string
		v    float32
	}{{"nan", float32(math.NaN())}, {"posinf", float32(math.Inf(1))}, {"neginf", float32(math.Inf(-1))}} {
		t.Run(c.name, func(t *testing.T) {
			s := benchSignal(24000, 24000)
			for i := 0; i < len(s); i += 977 {
				s[i] = c.v
			}
			img, err := Render(s, 24000, Options{XSize: 258, YSize: 129})
			if err != nil {
				t.Fatalf("Render: %v", err)
			}
			for i, p := range img.Pix {
				if int(p) >= len(img.Palette) {
					t.Fatalf("pixel %d index %d out of palette range %d", i, p, len(img.Palette))
				}
			}
		})
	}
}
