package sox

// Chrome layout constants from SoX spectrogram.c (stop()).
const (
	below         = 48 // rows under the raster (X labels, comment)
	left          = 58 // columns left of the raster (Y labels, axis name)
	between       = 37 // gap between raster and dBFS legend bar
	spectrumWidth = 14 // legend bar width
	right         = 35 // columns right of the legend bar (dB labels)
)

// Fixed palette indices (see palette.go fixedPalette).
const (
	backgroundIndex = 0
	textIndex       = 1
	labelsIndex     = 2
	gridIndex       = 3
)

// canvas is a bottom-up indexed pixel buffer mirroring SoX's pixel() macro:
// (0, 0) is the bottom-left of the final image.
type canvas struct {
	pix  []uint8
	cols int
}

func (c *canvas) set(x, y int, v uint8) { c.pix[y*c.cols+x] = v }

// printAt draws text left-to-right with (x, y) the top-left of the first
// glyph in bottom-up coordinates. Port of spectrogram.c print_at_().
func (c *canvas) printAt(x, y int, colour uint8, text string) {
	f := fontData()
	for k := 0; k < len(text); k++ {
		ch := text[k]
		if ch < ' ' || ch > '~' {
			ch = '~' + 1
		}
		pos := int(ch-' ') * fontY
		for i := 0; i < fontY; i++ {
			line := f[pos+i]
			for j := 0; j < fontX; j++ {
				if line&0x80 != 0 {
					c.set(x+j, y-i, colour)
				}
				line <<= 1
			}
		}
		x += fontAdvance
	}
}

// printUp draws text rotated 90 degrees counter-clockwise (reading upward).
func (c *canvas) printUp(x, y int, colour uint8, text string) {
	f := fontData()
	for k := 0; k < len(text); k++ {
		ch := text[k]
		if ch < ' ' || ch > '~' {
			ch = '~' + 1
		}
		pos := int(ch-' ') * fontY
		for i := 0; i < fontY; i++ {
			line := f[pos+i]
			for j := 0; j < fontX; j++ {
				if line&0x80 != 0 {
					c.set(x+i, y+j, colour)
				}
				line <<= 1
			}
		}
		y += fontAdvance
	}
}
