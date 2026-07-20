package sox

import (
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
//
// Every call allocates its own buffers, which makes it safe for concurrent use
// and the right choice for one-off renders. To render many images at the same
// Options, see NewRenderer.
func Render(samples []float32, sampleRate float64, opt Options) (*image.Paletted, error) {
	// Checked before the Options are validated, so that the error a caller sees
	// for a bad sample rate does not depend on whether the Options are also
	// wrong. NewRenderer validates in the other order, so this ordering is a
	// contract of the free function rather than an accident of the delegation;
	// TestRenderErrorPrecedence pins it.
	if err := checkSampleRate(sampleRate); err != nil {
		return nil, err
	}
	r, err := NewRenderer(opt)
	if err != nil {
		return nil, err
	}
	return r.Render(samples, sampleRate)
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
func WritePNG(path string, samples []float32, sampleRate float64, opt Options) error {
	img, err := Render(samples, sampleRate, opt)
	if err != nil {
		return err
	}
	return encodePNG(path, img)
}

// encodePNG is the write half of WritePNG; see its doc comment for why the
// write goes via a temporary file.
func encodePNG(path string, img *image.Paletted) (err error) {
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
