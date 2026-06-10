package sox

import (
	"fmt"
	"image"
	"image/png"
	"os"
)

// Render computes the spectrogram raster for mono `samples` at `sampleRate`
// (Hz) and returns it as an *image.Paletted with SoX's exact palette. The image
// is oriented like SoX: low frequency at the bottom row, time left-to-right,
// width = columns produced (capped at the effective x_size), height = rows
// (dft_size/2 + 1). Equivalent to `sox <in> -n spectrogram -r`.
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

	a := newAnalyzer(dft, rows, step, blocks, norm, -o.Gain, o.DBRange, ws, xSize)
	cols := a.run(samples)

	autogain := 0.0
	if o.Normalize {
		autogain = -a.max
	}

	pal := makePalette(o)
	img := image.NewPaletted(image.Rect(0, 0, cols, rows), pal)
	for col := 0; col < cols; col++ {
		for row := 0; row < rows; row++ {
			v := float64(a.dBfs[col*rows+row]) + autogain
			idx := colourIndex(o, v)
			// Flip vertically: spectrogram row 0 (DC) -> bottom image row.
			img.SetColorIndex(col, rows-1-row, uint8(idx))
		}
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
