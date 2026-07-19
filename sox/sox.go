package sox

import (
	"fmt"
	"image"
	"image/png"
	"os"
)

// Render computes the SoX-compatible spectrogram image for mono `samples` at
// `sampleRate`. By default it produces the full SoX PNG layout: raster plus
// axes, tick labels, dBFS legend, footer comment, and optional title,
// equivalent to `sox <in> -n spectrogram`. With opt.Raw it renders only the
// raster (`sox ... spectrogram -r`), oriented like SoX: low frequency at the
// bottom row, time left-to-right.
func Render(samples []float32, sampleRate float64, opt Options) (*image.Paletted, error) {
	if !(sampleRate > 0) { // also rejects NaN
		return nil, fmt.Errorf("sox: sampleRate must be positive, got %g", sampleRate)
	}
	o := normalize(opt)
	if err := validate(o); err != nil {
		return nil, err
	}

	dft, rows := deriveDFTSize(o)
	duration := float64(len(samples)) / sampleRate
	xSize, pps := resolveTimeAxis(o, duration)

	ws := newWindowState(dft, o.Window)
	actual := makeWindow(ws, 0)
	step, blocks, norm := stepSizing(actual, dft, sampleRate, pps, o.SlackOverlap)

	a, err := newAnalyzer(dft, rows, step, blocks, norm, -o.Gain, o.DBRange, ws, xSize)
	if err != nil {
		return nil, err
	}
	cols := a.run(samples)

	autogain := 0.0
	if o.Normalize {
		autogain = -a.max
	}

	colsTotal, rowsTotal := cols, rows
	rasterX, rasterY := 0, 0
	if !o.Raw {
		colsTotal, rowsTotal = chromeDims(cols, rows, o.Title)
		rasterX, rasterY = left, below
	}

	// Draw in SoX's bottom-up coordinates, then blit flipped.
	cv := &canvas{pix: make([]uint8, colsTotal*rowsTotal), cols: colsTotal}
	for col := 0; col < cols; col++ {
		for row := 0; row < rows; row++ {
			v := float64(a.dBfs[col*rows+row]) + autogain
			cv.set(rasterX+col, rasterY+row, uint8(colourIndex(o, v)))
		}
	}
	if !o.Raw {
		drawChrome(cv, chromeParams{
			rasterCols: cols,
			rasterRows: rows,
			colsTotal:  colsTotal,
			rowsTotal:  rowsTotal,
			secs:       float64(cols) * float64(step) * float64(blocks) / sampleRate,
			sampleRate: sampleRate,
			autogain:   autogain,
			o:          o,
		})
	}

	pal := makePalette(o)
	img := image.NewPaletted(image.Rect(0, 0, colsTotal, rowsTotal), pal)
	for y := 0; y < rowsTotal; y++ {
		copy(img.Pix[y*img.Stride:y*img.Stride+colsTotal],
			cv.pix[(rowsTotal-1-y)*colsTotal:(rowsTotal-y)*colsTotal])
	}
	return img, nil
}

// WritePNG renders and encodes to path with the stdlib PNG encoder.
func WritePNG(path string, samples []float32, sampleRate float64, opt Options) error {
	img, err := Render(samples, sampleRate, opt)
	if err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(f, img); err != nil {
		f.Close()
		os.Remove(path) // don't leave a partial/corrupt file behind
		return err
	}
	return f.Close()
}
