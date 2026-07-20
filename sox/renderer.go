package sox

import (
	"fmt"
	"image"
	"image/color"
)

// Renderer renders repeatedly at one fixed Options, holding on to the buffers a
// render needs instead of rebuilding them per call: the analysis output, the
// schedule, the per-worker DSP state and transform plans, the canvas, and the
// image's pixels. A consumer producing many images at a single preset, which is
// the usual way this package is used, pays for those once rather than per image.
// See the README for the measured figures.
//
// It is not the whole of a render's allocation. Chrome formats its tick labels
// per image and the fan-out allocates a closure per worker, neither of which
// this touches; the reuse is about the large buffers and the transform plan. Its
// clearest effect is on the collector rather than on any one render, since those
// buffers stop being garbage.
//
// It is strictly an optimisation: for any given clip it produces exactly the
// image the free Render produces, and Render is implemented on top of it.
//
// Two things are traded for that. The returned image's pixels are reused, so a
// caller must treat an image as invalid once it calls Render again; WritePNG
// encodes before returning and so never exposes this. And a Renderer is
// stateful, so it must not be used from more than one goroutine at a time: make
// one per worker goroutine, as with mel.Generator and fft.Plan. The free Render
// stays safe for concurrent use.
//
// The buffers are grow-only and grow geometrically, so a Renderer holds roughly
// two canvases plus the analysis output and the per-worker state for its
// lifetime, and up to twice that where a buffer last doubled (a few megabytes at
// 1026x513, more with Normalize, which adds a float32 buffer four times the size
// of the palette indices). Drop the Renderer to release them.
type Renderer struct {
	o    Options // normalized and validated
	dft  int
	rows int

	// Fixed by Options, so computed once: the palette every returned image
	// shares, the palette's spectrum length, and the raw sum of the full-length
	// window, which is what sizes the hop. The worker count is deliberately not
	// here; see Render.
	pal            color.Palette
	spectrumPoints int
	windowSum      float64

	scratch analysisScratch
	cv      canvas
	pix     []uint8
}

// NewRenderer validates opt and computes the state that depends on it alone. It
// returns the same errors Render would for the same Options, just at
// construction rather than on the first clip.
//
// The buffers themselves are allocated on first use, since their size depends on
// the clip, so the first Render through a new Renderer costs what a free Render
// costs and later ones do not.
func NewRenderer(opt Options) (*Renderer, error) {
	o := normalize(opt)
	if err := validate(o); err != nil {
		return nil, err
	}
	dft, rows := deriveDFTSize(o)
	return &Renderer{
		o: o, dft: dft, rows: rows,
		pal:            makePalette(o),
		spectrumPoints: spectrumPoints(o),
		// The full-length window is only needed for its sum; each analysis
		// goroutine reshapes its own copy from there.
		windowSum: makeWindow(newWindowState(dft, o.Window), 0),
	}, nil
}

// Render computes the spectrogram image for mono `samples` at `sampleRate`,
// reusing the buffers from the previous call.
//
// The returned image aliases those buffers, so it must be treated as invalid
// once Render is called again on this Renderer. Note that the invalidation is
// not reliably observable: when the next clip needs a larger buffer the old
// pixels survive untouched, so a caller that wrongly holds on to an image can
// pass its own tests on one ordering of clips and produce corrupt output on
// another. Copy the image, or use WritePNG.
//
// The header itself is fresh each call, so an image held from an earlier call
// keeps its own Rect and Stride. Its Palette is shared by every image this
// Renderer returns and must not be modified.
func (r *Renderer) Render(samples []float32, sampleRate float64) (*image.Paletted, error) {
	if err := checkSampleRate(sampleRate); err != nil {
		return nil, err
	}
	o := r.o
	duration := float64(len(samples)) / sampleRate
	xSize, pps := resolveTimeAxis(o, duration)
	step, blocks, norm := stepSizing(r.windowSum, r.dft, sampleRate, pps, o.SlackOverlap)

	// Resolved per call rather than cached, so that a long-lived Renderer
	// tracks GOMAXPROCS the way the free Render does. Go adjusts it at runtime
	// from the cgroup CPU limit, so a container that is resized would otherwise
	// leave this Renderer fanning out over a stale core count for its lifetime.
	workers := resolveWorkers(o.Workers)

	a, err := analyze(analyzerOpts{
		dftSize: r.dft, rows: r.rows,
		stepSize: step, blockSteps: blocks, blockNorm: norm,
		gain: -o.Gain, dBRange: o.DBRange,
		spectrumPoints: r.spectrumPoints,
		xSize:          xSize,
		normalize:      o.Normalize,
		window:         o.Window,
		workers:        workers,
	}, samples, &r.scratch)
	if err != nil {
		return nil, fmt.Errorf("sox: analyzing at DFT size %d: %w", r.dft, err)
	}
	cols := a.cols

	autogain := 0.0
	if o.Normalize {
		autogain = -a.max
		a.quantise(autogain)
	}

	colsTotal, rowsTotal := cols, r.rows
	rasterX, rasterY := 0, 0
	if !o.Raw {
		colsTotal, rowsTotal = chromeDims(cols, r.rows, o.Title)
		rasterX, rasterY = left, below
	}

	// Draw in SoX's bottom-up coordinates, then blit flipped.
	n := colsTotal * rowsTotal
	var fresh bool
	r.cv.pix, fresh = growFresh(r.cv.pix, n)
	r.cv.cols = colsTotal
	if !o.Raw && !fresh {
		// Chrome paints only its text, ticks and border, and takes the rest of
		// the canvas already being the background colour (palette index 0). On a
		// reused buffer that has to be made true again, or the previous image's
		// chrome ghosts through. A buffer grow just allocated is already zeroed,
		// and with Raw the blit below covers every byte, so both of those skip
		// the clear rather than merely not needing it.
		clear(r.cv.pix)
	}
	rasterBlit{
		dst: r.cv.pix, src: a.idx,
		cols: cols, rows: r.rows,
		rasterX: rasterX, rasterY: rasterY, colsTotal: colsTotal,
	}.run(workers)
	if !o.Raw {
		drawChrome(&r.cv, chromeParams{
			rasterCols: cols,
			rasterRows: r.rows,
			colsTotal:  colsTotal,
			rowsTotal:  rowsTotal,
			secs:       float64(cols) * float64(step) * float64(blocks) / sampleRate,
			sampleRate: sampleRate,
			autogain:   autogain,
			o:          o,
		})
	}

	r.pix = grow(r.pix, n)
	img := &image.Paletted{
		// Capped to its length, as image.NewPaletted's is: a grown buffer can
		// carry spare capacity, and without the cap a caller's append would
		// write into this Renderer's live pixels instead of a copy.
		Pix:     r.pix[:n:n],
		Stride:  colsTotal,
		Rect:    image.Rect(0, 0, colsTotal, rowsTotal),
		Palette: r.pal,
	}
	for y := 0; y < rowsTotal; y++ {
		copy(img.Pix[y*img.Stride:y*img.Stride+colsTotal],
			r.cv.pix[(rowsTotal-1-y)*colsTotal:(rowsTotal-y)*colsTotal])
	}
	return img, nil
}

// WritePNG renders and encodes to path exactly as the package-level WritePNG
// does, reusing this Renderer's buffers. The image never escapes, so the
// aliasing Render documents is invisible to the caller and this is the safest
// way to use a Renderer.
func (r *Renderer) WritePNG(path string, samples []float32, sampleRate float64) error {
	img, err := r.Render(samples, sampleRate)
	if err != nil {
		return err
	}
	return encodePNG(path, img)
}

// checkSampleRate is shared by both entry points so that the message a caller
// sees cannot come to depend on which one they used.
func checkSampleRate(sampleRate float64) error {
	if !(sampleRate > 0) { // also rejects NaN
		return fmt.Errorf("sox: sampleRate must be positive, got %g", sampleRate)
	}
	return nil
}

// growFresh returns a slice of exactly n elements, reusing b's array when it is
// already large enough, and reports whether it had to allocate.
//
// The contents are not cleared on reuse: every caller here writes all n elements
// before reading any, and the one that does not (the canvas, which chrome only
// partly paints) clears explicitly. The `fresh` result is what lets it skip even
// that, since a slice make just returned is already zeroed.
//
// Growth is geometric rather than an exact fit. The buffers are sized by the
// column count, which tracks clip duration when the caller drives the time axis
// with PixelsPerSec, so growing tightly would reallocate on every clip slightly
// longer than the last, which is the reuse this package exists to provide. A nil
// slice always allocates, so that a zero-length result is non-nil, matching what
// image.NewPaletted produced before.
func growFresh[S ~[]E, E any](b S, n int) (S, bool) {
	if b == nil || cap(b) < n {
		return make(S, n, max(n, 2*cap(b))), true
	}
	return b[:n], false
}

// grow is growFresh for the callers that have nothing to clear.
func grow[S ~[]E, E any](b S, n int) S {
	s, _ := growFresh(b, n)
	return s
}
