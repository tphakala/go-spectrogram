package sox

import (
	"image/color"
	"os/exec"
	"testing"
)

func TestColourIndex(t *testing.T) {
	o := normalize(Options{}) // dBRange 120, quant 249 -> sp = 251
	sp := spectrumPoints(o)
	if sp != 251 {
		t.Fatalf("spectrumPoints = %d, want 251", sp)
	}
	if got := colourIndex(o, -1000); got != fixedPalette+0 {
		t.Errorf("below range: %d, want %d", got, fixedPalette)
	}
	if got := colourIndex(o, 0); got != fixedPalette+sp-1 {
		t.Errorf("at 0 dB: %d, want %d", got, fixedPalette+sp-1)
	}
	if got := colourIndex(o, 5); got != fixedPalette+sp-1 {
		t.Errorf("above 0 dB: %d, want %d", got, fixedPalette+sp-1)
	}
	// At exactly -dBRange => c = 1.
	if got := colourIndex(o, -120); got != fixedPalette+1 {
		t.Errorf("at -120 dB: %d, want %d", got, fixedPalette+1)
	}
}

func TestSpectrumPointsAltCap(t *testing.T) {
	o := normalize(Options{AltPalette: true})
	if sp := spectrumPoints(o); sp != altPaletteLen {
		t.Errorf("alt spectrumPoints = %d, want %d", sp, altPaletteLen)
	}
}

func TestMakePaletteLength(t *testing.T) {
	if got := len(makePalette(normalize(Options{}))); got != fixedPalette+251 {
		t.Errorf("default palette len = %d, want %d", got, fixedPalette+251)
	}
	if got := len(makePalette(normalize(Options{AltPalette: true}))); got != fixedPalette+altPaletteLen {
		t.Errorf("alt palette len = %d, want %d", got, fixedPalette+altPaletteLen)
	}
}

func TestMakePaletteFixedEntries(t *testing.T) {
	p := makePalette(normalize(Options{})) // dark bg
	want := []color.RGBA{
		{0, 0, 0, 255},       // Background black
		{255, 255, 255, 255}, // Text white
		{0xbf, 0xbf, 0xbf, 255},
		{0x7f, 0x7f, 0x7f, 255},
	}
	for i, w := range want {
		if p[i] != w {
			t.Errorf("fixed[%d]=%v want %v", i, p[i], w)
		}
	}
}

// Oracle: compare our default palette to the PLTE chunk SoX writes.
func TestMakePaletteMatchesSoXPLTE(t *testing.T) {
	requireSox(t)
	cases := []struct {
		name string
		opt  Options
		args []string
	}{
		{"default", Options{}, nil},
		{"mono", Options{Monochrome: true}, []string{"-m"}},
		{"high", Options{HighColour: true}, []string{"-h"}},
		{"alt", Options{AltPalette: true}, []string{"-A"}},
		{"light", Options{LightBackground: true}, []string{"-l"}},
		{"perm3", Options{Perm: 3}, []string{"-p", "3"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ref := soxPLTE(t, c.args)        // [][3]byte from sox PNG
			got := makePalette(normalize(c.opt))
			if len(got) != len(ref) {
				t.Fatalf("palette len %d != sox %d", len(got), len(ref))
			}
			for i := range ref {
				r, g, b, _ := got[i].RGBA()
				gr, gg, gb := byte(r>>8), byte(g>>8), byte(b>>8)
				if gr != ref[i][0] || gg != ref[i][1] || gb != ref[i][2] {
					t.Errorf("entry %d = %02x%02x%02x, sox %02x%02x%02x",
						i, gr, gg, gb, ref[i][0], ref[i][1], ref[i][2])
				}
			}
		})
	}
}

func requireSox(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("sox"); err != nil {
		t.Skip("sox binary not found; skipping oracle test")
	}
}
