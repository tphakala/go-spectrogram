package sox

import "testing"

// glyphPixels returns the set of lit (x,y) for glyph ch drawn by the C
// algorithm: row i bit j (from 0x80) lit means (x+j, y-i) for printAt and
// (x+i, y+j) for printUp.
func glyphPixels(ch byte, x, y int, up bool) map[[2]int]bool {
	f := fontData()
	want := map[[2]int]bool{}
	pos := int(ch-' ') * fontY
	for i := 0; i < fontY; i++ {
		line := f[pos+i]
		for j := 0; j < fontX; j++ {
			if line&(0x80>>j) != 0 {
				if up {
					want[[2]int{x + i, y + j}] = true
				} else {
					want[[2]int{x + j, y - i}] = true
				}
			}
		}
	}
	return want
}

func checkCanvas(t *testing.T, c *canvas, rows int, want map[[2]int]bool, colour uint8) {
	t.Helper()
	for y := 0; y < rows; y++ {
		for x := 0; x < c.cols; x++ {
			got := c.pix[y*c.cols+x]
			if want[[2]int{x, y}] {
				if got != colour {
					t.Errorf("pixel (%d,%d) = %d, want %d", x, y, got, colour)
				}
			} else if got != 0 {
				t.Errorf("pixel (%d,%d) = %d, want 0", x, y, got)
			}
		}
	}
}

func TestPrintAt(t *testing.T) {
	c := &canvas{pix: make([]uint8, 40*40), cols: 40}
	c.printAt(10, 20, textIndex, "0!")
	want := glyphPixels('0', 10, 20, false)
	for k, v := range glyphPixels('!', 10+fontAdvance, 20, false) {
		want[k] = v
	}
	checkCanvas(t, c, 40, want, textIndex)
}

func TestPrintUp(t *testing.T) {
	c := &canvas{pix: make([]uint8, 40*40), cols: 40}
	c.printUp(5, 8, textIndex, "0!")
	want := glyphPixels('0', 5, 8, true)
	for k, v := range glyphPixels('!', 5, 8+fontAdvance, true) {
		want[k] = v
	}
	checkCanvas(t, c, 40, want, textIndex)
}

func TestPrintAtFallbackGlyph(t *testing.T) {
	c := &canvas{pix: make([]uint8, 40*40), cols: 40}
	c.printAt(10, 20, textIndex, "\x7f") // out of range -> fallback glyph 95
	want := glyphPixels('~'+1, 10, 20, false)
	checkCanvas(t, c, 40, want, textIndex)
}

func TestAxisScale(t *testing.T) {
	cases := []struct {
		to       float64
		maxSteps int
		step     int
		limit    float64
		prefix   string
	}{
		{0.5, 29, 200, 5000, "m"},
		{2.323, 29, 1, 23.23, ""},
		{10, 29, 5, 100, ""},
		{60, 29, 50, 600, ""},
		{120.7, 29, 50, 1207, ""},
		{22050, 28, 10, 220.5, "k"},
		{4000, 28, 2, 40, "k"},
		{24000, 28, 10, 240, "k"},
		{11025, 28, 5, 110.25, "k"},
		{0.05, 10, 50, 500, "m"},
		{1, 0, 10, 1, ""},
		{3.9999, 29, 2, 39.998999999999995, ""},
		{0.0001, 28, 50, 1000, "u"},
	}
	for _, c := range cases {
		step, limit, prefix := axisScale(c.to, c.maxSteps)
		if step != c.step || prefix != c.prefix {
			t.Errorf("axisScale(%g, %d) = step %d prefix %q, want %d %q",
				c.to, c.maxSteps, step, prefix, c.step, c.prefix)
		}
		if diff := limit - c.limit; diff > 1e-9 || diff < -1e-9 {
			t.Errorf("axisScale(%g, %d) limit = %.17g, want %.17g",
				c.to, c.maxSteps, limit, c.limit)
		}
	}
}
