package sox

import (
	"fmt"
	"math"
)

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

// chromeParams carries everything drawChrome needs from Render.
type chromeParams struct {
	rasterCols int
	rasterRows int
	colsTotal  int
	rowsTotal  int
	secs       float64 // seconds spanned by the raster: cols*step*blocks/rate
	sampleRate float64
	dBRange    int
	gain       int     // Options.Gain (sox -Z, un-negated)
	autogain   float64 // -max when Normalize, else 0
	title      string
	comment    string
	noAxes     bool
	o          Options // normalized options, for colourIndex
}

// drawChrome ports the !p->raw branch of spectrogram.c stop() (mono path).
// All coordinates are bottom-up canvas coordinates.
func drawChrome(c *canvas, p chromeParams) {
	tickLen := 3
	if p.noAxes {
		tickLen = 2
	}

	if !p.noAxes { // grid border around the raster
		for j := 0; j < p.rasterRows; j++ {
			c.set(left-1, below+j, gridIndex)
			c.set(left+p.rasterCols, below+j, gridIndex)
		}
		for i := -1; i <= p.rasterCols; i++ {
			c.set(left+i, below-1, gridIndex)
			c.set(left+i, below+p.rasterRows, gridIndex)
		}
	}

	if p.title != "" {
		if w := len(p.title) * fontAdvance; w < p.colsTotal+1 {
			c.printAt((p.colsTotal-w)/2, p.rowsTotal-fontY, textIndex, p.title)
		}
	}
	if len(p.comment)*fontAdvance < p.colsTotal+1 {
		c.printAt(1, fontY, textIndex, p.comment)
	}

	// X axis (time).
	step, limit, prefix := axisScale(p.secs, p.rasterCols/(fontAdvance*9/2))
	label := fmt.Sprintf("Time (%ss)", prefix)
	c.printAt(left+(p.rasterCols-fontAdvance*len(label))/2, 24, textIndex, label)
	for i := 0; float64(i) <= limit; i += step {
		x := 0
		if limit != 0 {
			x = int(float64(i)/limit*float64(p.rasterCols) + .5)
		}
		for y := 0; y < tickLen; y++ {
			c.set(left-1+x, below-1-y, gridIndex)
			c.set(left-1+x, below+p.rasterRows+y, gridIndex)
		}
		if step == 5 && i%10 != 0 {
			continue
		}
		text := fmt.Sprintf("%.6g", 0.1*float64(i))
		lx := left + x - 3*len(text)
		c.printAt(lx, below-6, labelsIndex, text)
		c.printAt(lx, below+p.rasterRows+14, labelsIndex, text)
	}

	// Y axis (frequency).
	step, limit, prefix = axisScale(p.sampleRate/2, (p.rasterRows-1)/((fontY*3+1)>>1))
	label = fmt.Sprintf("Frequency (%sHz)", prefix)
	c.printUp(10, below+(p.rasterRows-fontAdvance*len(label))/2, textIndex, label)
	for i := 0; float64(i) <= limit; i += step {
		y := 0
		if limit != 0 {
			y = int(float64(i)/limit*float64(p.rasterRows-1) + .5)
		}
		for x := 0; x < tickLen; x++ {
			c.set(left-1-x, below+y, gridIndex)
			c.set(left+p.rasterCols+x, below+y, gridIndex)
		}
		if step == 5 && i%10 != 0 {
			continue
		}
		ltext, rtext := "   DC", "DC"
		if i != 0 {
			ltext = fmt.Sprintf("%5.6g", 0.1*float64(i))
			rtext = fmt.Sprintf("%.6g", 0.1*float64(i))
		}
		c.printAt(left-4-fontAdvance*5, below+y+5, labelsIndex, ltext)
		c.printAt(left+p.rasterCols+6, below+y+5, labelsIndex, rtext)
	}

	// Z axis (dBFS legend).
	k := p.rasterRows
	if k > 400 {
		k = 400
	}
	zbase := below + (p.rasterRows-k)/2
	c.printAt(p.colsTotal-right-2-fontAdvance, zbase-13, textIndex, "dBFS")
	for j := 0; j < k; j++ {
		b := uint8(colourIndex(p.o, float64(p.dBRange)*(float64(j)/float64(k-1)-1)))
		for i := 0; i < spectrumWidth; i++ {
			c.set(p.colsTotal-right-1-i, zbase+j, b)
		}
	}
	zstep := 10 * int(math.Ceil(float64(p.dBRange)/10*(fontY+2)/float64(k-1)))
	for i := 0; i <= p.dBRange; i += zstep {
		y := int(float64(i)/float64(p.dBRange)*float64(k-1) + .5)
		text := fmt.Sprintf("%+d", i+p.gain-p.dBRange-int(p.autogain+.5))
		c.printAt(p.colsTotal-right+1, zbase+y+5, labelsIndex, text)
	}
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
