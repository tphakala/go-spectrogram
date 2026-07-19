package sox

import (
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"sync"
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

	workers := resolveWorkers(o.Workers)
	// The full-length window is only needed for its sum, which sizes the hop;
	// each analysis goroutine reshapes its own copy from there.
	actual := makeWindow(newWindowState(dft, o.Window), 0)
	step, blocks, norm := stepSizing(actual, dft, sampleRate, pps, o.SlackOverlap)

	a, err := analyze(analyzerOpts{
		dftSize: dft, rows: rows,
		stepSize: step, blockSteps: blocks, blockNorm: norm,
		gain: -o.Gain, dBRange: o.DBRange,
		spectrumPoints: spectrumPoints(o),
		xSize:          xSize,
		normalize:      o.Normalize,
		window:         o.Window,
		workers:        workers,
	}, samples)
	if err != nil {
		return nil, fmt.Errorf("sox: analyzing at DFT size %d: %w", dft, err)
	}
	cols := a.cols

	autogain := 0.0
	if o.Normalize {
		autogain = -a.max
		a.quantise(autogain)
	}

	colsTotal, rowsTotal := cols, rows
	rasterX, rasterY := 0, 0
	if !o.Raw {
		colsTotal, rowsTotal = chromeDims(cols, rows, o.Title)
		rasterX, rasterY = left, below
	}

	// Draw in SoX's bottom-up coordinates, then blit flipped.
	cv := &canvas{pix: make([]uint8, colsTotal*rowsTotal), cols: colsTotal}
	rasterBlit{
		dst: cv.pix, src: a.idx,
		cols: cols, rows: rows,
		rasterX: rasterX, rasterY: rasterY, colsTotal: colsTotal,
	}.run(workers)
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

// rasterBlit transposes the analysis output, which is column-major, into the
// canvas, which is row-major.
type rasterBlit struct {
	dst, src         []uint8
	cols, rows       int
	rasterX, rasterY int
	colsTotal        int
}

// blitTile is the side of the transpose that is walked in tiles. Walking the
// source naively streams one column against a destination stride of colsTotal,
// which evicts the destination from cache once per column at larger sizes;
// tiling keeps both sides resident.
const blitTile = 64

// run performs the transpose, spread over up to workers goroutines.
//
// The split is by column tile, so each worker writes a disjoint span of every
// row it touches and the shared destination needs no synchronisation. It is
// worth parallelising at all because the analysis on either side of it already
// is: left serial, moving a couple of megabytes through a strided read is a
// quarter of the wall time of a large render.
func (b rasterBlit) run(workers int) {
	tiles := (b.cols + blitTile - 1) / blitTile
	workers = min(max(workers, 1), tiles)
	if workers <= 1 {
		b.tiles(0, tiles)
		return
	}
	var wg sync.WaitGroup
	per := (tiles + workers - 1) / workers
	for w := range workers {
		from, to := w*per, min((w+1)*per, tiles)
		if from >= to {
			break
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			b.tiles(from, to)
		}()
	}
	wg.Wait()
}

// tiles transposes the column tiles in [fromTile, toTile).
func (b rasterBlit) tiles(fromTile, toTile int) {
	for t := fromTile; t < toTile; t++ {
		colBase := t * blitTile
		colEnd := min(colBase+blitTile, b.cols)
		for rowBase := 0; rowBase < b.rows; rowBase += blitTile {
			rowEnd := min(rowBase+blitTile, b.rows)
			// row-then-col inside the tile so the destination is written
			// sequentially; the strided source reads stay resident for a tile.
			for row := rowBase; row < rowEnd; row++ {
				// Slice to exactly the tile's span so the WRITE is bounds-check
				// free; ranging over it is what proves dst[i] in range. The
				// strided source read still carries one check per pixel, which
				// Go has no way to express away.
				start := (b.rasterY+row)*b.colsTotal + b.rasterX + colBase
				dst := b.dst[start : start+colEnd-colBase]
				col := b.src[colBase*b.rows+row:]
				for i := range dst {
					dst[i] = col[i*b.rows]
				}
			}
		}
	}
}
