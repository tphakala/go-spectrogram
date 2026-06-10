package sox

import "math"

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

// set writes v at column x, row y in bottom-up coordinates (row 0 is the
// bottom of the final image).
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

// chromeDims returns the full-image size for a given raster size (mono:
// c_rows == raster rows). From stop(): rows = below + c_rows + 30 + 20*title,
// cols = left + cols + between + spectrum_width + right.
func chromeDims(rasterCols, rasterRows int, title string) (cols, rows int) {
	cols = left + rasterCols + between + spectrumWidth + right
	rows = below + rasterRows + 30
	if title != "" {
		rows += 20
	}
	return cols, rows
}

// axisScale ports SoX's axis(): pick a tick step for the range [0, to] with
// at most maxSteps ticks. When maxSteps > 0, step and limit are in tenths of
// the displayed unit (after applying the SI prefix); prefix is "" or one of
// p n u m k M G T P E. When maxSteps == 0 the scaling branch is skipped and
// limit equals the original to.
func axisScale(to float64, maxSteps int) (step int, limit float64, prefix string) {
	scale := 1.0
	fstep := math.Max(1, 10*to)
	prefixNum := 0
	if maxSteps != 0 {
		log10v := math.Inf(1)
		to *= 10
		minStep := to / float64(maxSteps)
		for i := 5; i > 0; i >>= 1 {
			// Snap near-integer log10 values to the integer before ceil, matching
			// C library log10 behaviour (e.g. log10(0.1) == -1 in glibc but
			// slightly above -1 in Go, causing ceil to return 0 instead of -1).
			l := math.Log10(minStep * float64(i))
			if r := math.Round(l); math.Abs(l-r) < 1e-9 {
				l = r
			}
			if try := math.Ceil(l); try <= log10v {
				log10v = try
				fstep = math.Pow(10, log10v) / float64(i)
				if i > 1 {
					log10v--
				}
			}
		}
		prefixNum = int(math.Floor(log10v / 3))
		scale = math.Pow(10, -3*float64(prefixNum))
	}
	// C: "pnum-kMGTPE" + prefix_num + (prefix_num ? 4 : 11). Index 11 is the
	// NUL terminator, so no prefix when prefixNum == 0.
	const prefixes = "pnum-kMGTPE"
	if prefixNum != 0 {
		if idx := prefixNum + 4; idx >= 0 && idx < len(prefixes) {
			prefix = prefixes[idx : idx+1]
		}
	}
	return int(fstep*scale + .5), to * scale, prefix
}
