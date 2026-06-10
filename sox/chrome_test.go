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
