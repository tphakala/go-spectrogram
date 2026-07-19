package sox

import (
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
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
		return nil, fmt.Errorf("sox: building the analyzer for DFT size %d: %w", dft, err)
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
	// Resolve the palette constants once: they are fixed for the whole raster,
	// and this loop runs once per pixel.
	sp, dbRange := spectrumPoints(o), float64(o.DBRange)
	// dBfs is column-major but the canvas is row-major, so this is a transpose.
	// Walking it naively streams one source column against a destination stride
	// of colsTotal, which evicts the destination from cache once per column at
	// larger sizes. Tiling keeps both sides of the transpose resident.
	const tile = 64
	for colBase := 0; colBase < cols; colBase += tile {
		colEnd := min(colBase+tile, cols)
		for rowBase := 0; rowBase < rows; rowBase += tile {
			rowEnd := min(rowBase+tile, rows)
			// row-then-col inside the tile so the destination is written
			// sequentially; the strided source reads stay in L1 for a 64x64 tile.
			for row := rowBase; row < rowEnd; row++ {
				// Slice to exactly the tile's span so the WRITE is bounds-check
				// free; ranging over it is what proves dst[i] in range. The
				// strided source read still carries one check per pixel, which
				// Go has no way to express away.
				start := (rasterY+row)*colsTotal + rasterX + colBase
				dst := cv.pix[start : start+colEnd-colBase]
				for i := range dst {
					v := float64(a.dBfs[(colBase+i)*rows+row]) + autogain
					dst[i] = uint8(colourIndexAt(v, sp, dbRange))
				}
			}
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
//
// The image is written to a unique temporary file in the destination directory
// and renamed into place, so path either does not exist yet or holds a complete
// PNG. Encoding a large spectrogram takes milliseconds and consumers commonly
// serve these files while they are being produced; writing in place would let a
// reader observe a truncated image.
//
// The rename is what provides atomicity. There is deliberately no fsync: it
// would cost more than the encode on slow storage, and it buys durability
// across a power cut rather than atomicity, which is not a guarantee the sox
// binary offers either.
//
// Windows note: os.Rename replaces an existing destination (Go passes
// MOVEFILE_REPLACE_EXISTING), so repeated writes to the same path work. It can
// however fail with a sharing violation if another process holds the
// destination open without FILE_SHARE_DELETE, where the previous truncate in
// place would have succeeded. That trade is deliberate: a reader that had the
// old file open keeps reading a complete image rather than watching one be
// overwritten underneath it.
func WritePNG(path string, samples []float32, sampleRate float64, opt Options) (err error) {
	img, err := Render(samples, sampleRate, opt)
	if err != nil {
		return err
	}

	// Same directory as the destination, so the rename cannot cross a
	// filesystem boundary. A unique name keeps concurrent writers of the same
	// destination from sharing a temporary.
	f, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".tmp*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	renamed := false
	defer func() {
		// Keyed on renamed, not on err: a panic in the encoder would leave err
		// nil and strand the temporary file otherwise.
		if !renamed {
			// Best-effort cleanup on the failure path; the caller already has
			// the error that got us here, so these add nothing.
			_ = f.Close()
			_ = os.Remove(tmp)
		}
	}()

	if err = png.Encode(f, img); err != nil {
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	// os.CreateTemp creates with 0600, which would silently break the common
	// deployment where another user or container serves the file. os.Create
	// gave 0666&^umask; this does not reproduce that exactly, it guarantees
	// group and world readability and owner write, which is what consumers
	// actually depend on.
	if err = os.Chmod(tmp, 0o644); err != nil {
		return err
	}
	if err = os.Rename(tmp, path); err != nil {
		return err
	}
	renamed = true
	return nil
}
