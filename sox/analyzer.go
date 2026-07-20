package sox

import (
	"fmt"
	"sync"

	"github.com/tphakala/go-spectrogram/internal/fft"
	"github.com/tphakala/simd/f64"
)

// analyzerOpts is what the analysis needs from the resolved Options and the
// geometry derived from them. It is a struct rather than a parameter list
// because that list had grown to ten positional values, most of them ints,
// where transposing two of them is a wrong image rather than a compile error.
type analyzerOpts struct {
	dftSize, rows        int
	stepSize, blockSteps int
	blockNorm            float64
	gain                 int // SoX p->gain = -opt.Gain
	dBRange              int
	spectrumPoints       int
	xSize                int
	normalize            bool // SoX -n
	window               WindowType
	workers              int // goroutines to spread the columns over; >= 1
}

// blockPlan is one DFT: where its frame starts in the padded input stream, and
// the window `end` SoX's state machine had reached when it ran.
//
// start is a position in the stream SoX effectively transforms, which is the
// input with zeros before the first sample and after the last: the leading pad
// is what centres the first frame, the trailing pad is what the drain flushes
// with. It is therefore signed, and can also run past the end of the input.
type blockPlan struct {
	start int
	end   int
}

// columnPlan is one output column: the run of blocks whose power it sums, and
// the normalisation SoX applies to that sum. norm is per column because the
// last one can cover fewer blocks than the rest and is rescaled to match.
type columnPlan struct {
	firstBlock int
	nBlocks    int
	norm       float64
}

// analysis is the finished spectrogram data for an image.
//
// Output is column-major (column c occupies [c*rows : (c+1)*rows], row 0 = DC)
// and takes one of two forms. By default each cell is resolved to a palette
// index in idx as its column is computed, while that column's dB values are
// still in L1: a quarter of the bytes of the dBFS form, and it leaves Render's
// raster pass a plain byte transpose rather than a strided float32 read with a
// quantisation per pixel.
//
// With -n normalize the mapping depends on the brightest cell in the whole
// image, which is not known until every column is done, so the raw dBFS values
// go to dBfs and quantise resolves them afterwards. max, the brightest pre-gain
// dBFS, is only maintained in that case; it stays at its -dBRange seed
// otherwise.
type analysis struct {
	opts analyzerOpts
	cols int

	idx  []uint8   // palette indices, default path
	dBfs []float32 // raw dBFS, -n normalize path
	max  float64
}

// analysisScratch is everything the analysis allocates that a Renderer can
// hand back on the next call: the output buffers, the schedule, the transform
// plan and the per-worker DSP state.
//
// The buffers are grow-only. Their sizes track the column count, which varies
// with clip duration, so a Renderer fed one long clip and then many short ones
// holds the long clip's footprint until it is dropped. That is the trade a
// reuse API exists to make, and the alternative (shrinking, or sizing to
// Options.XSize up front) either reintroduces the allocation or pessimises the
// common fixed-preset case.
type analysisScratch struct {
	a       analysis
	idx     []uint8
	dBfs    []float32
	blocks  []blockPlan
	columns []columnPlan
	crs     []*columnRenderer
	plan    *fft.Plan
}

// growU8 returns a slice of exactly n bytes, reusing b's array when it already
// has the capacity. The contents are not cleared: every caller here writes all
// n bytes before reading any, and the one place that does not (the canvas,
// which chrome only partly paints) clears explicitly.
func growU8(b []uint8, n int) []uint8 {
	if cap(b) < n {
		return make([]uint8, n)
	}
	return b[:n]
}

func growF32(b []float32, n int) []float32 {
	if cap(b) < n {
		return make([]float32, n)
	}
	return b[:n]
}

// analyze renders every column of the spectrogram for samples.
//
// It runs in two phases. The first walks SoX's streaming state machine to
// decide what each column is made of; the second computes the columns, over as
// many goroutines as opts.workers allows. Splitting it this way is what makes
// the transform parallelisable at all: SoX's accumulation is sequential state
// (a sliding buffer, a window that reshapes at both ends of the clip, a
// truncation rule, and a renormalised final column), so the schedule is derived
// by running that state machine rather than by re-deriving closed forms for it,
// which is how a parallel port drifts from the original. What comes out is a
// plan whose columns are independent and can be computed in any order.
func analyze(o analyzerOpts, samples []float32, s *analysisScratch) (*analysis, error) {
	if s.plan == nil {
		// dftSize is a power of two by construction (validate() rejects YSize
		// values that do not yield one), so this only fails on a programming
		// error. It is built once per scratch: dftSize derives from Options
		// alone, which a Renderer fixes for its lifetime.
		plan, err := fft.NewPlan(o.dftSize)
		if err != nil {
			return nil, err
		}
		s.plan = plan
	}
	plan := s.plan
	// deriveDFTSize always returns rows == dftSize/2+1, which is exactly the
	// plan's bin count. Assert it anyway: rows sizes every per-column buffer
	// here, and the simd reductions in finishColumn silently process only
	// min(len(dst), len(src)) elements, so a mismatch would quietly truncate
	// each column rather than fail.
	if o.rows != plan.NumBins() {
		return nil, fmt.Errorf("sox: rows %d does not match DFT bin count %d for size %d",
			o.rows, plan.NumBins(), o.dftSize)
	}

	// [:0] rather than fresh slices: schedule appends, so reusing them unsliced
	// would append this clip's schedule after the previous one's.
	s.blocks, s.columns = schedule(o, len(samples), s.blocks[:0], s.columns[:0])
	blocks, columns := s.blocks, s.columns

	// Assigning the whole struct is what resets it, max included: a maximum
	// carried over from a louder clip would render this one dark under -n.
	a := &s.a
	*a = analysis{opts: o, cols: len(columns), max: -float64(o.dBRange)}
	n := a.cols * o.rows
	// idx is sized on both paths. The -n path fills it in quantise rather than
	// in emit, but it is the same buffer and the same layout either way.
	s.idx = growU8(s.idx, n)
	a.idx = s.idx
	if o.normalize {
		s.dBfs = growF32(s.dBfs, n)
		a.dBfs = s.dBfs
	}
	if a.cols == 0 {
		return a, nil
	}

	workers := min(max(o.workers, 1), a.cols)
	// Grown to the largest worker count seen, never clamped down: a Renderer
	// whose first clip was short enough to need one worker must still fan out
	// on the next one.
	for len(s.crs) < workers {
		// The first worker takes the plan itself; the rest clone it, sharing its
		// twiddle tables and allocating only their own scratch.
		p := plan
		if len(s.crs) > 0 {
			p = plan.Clone()
		}
		s.crs = append(s.crs, newColumnRenderer(o, p))
	}
	rs := s.crs[:workers]
	for _, r := range rs {
		r.reset(a)
	}

	if workers == 1 {
		rs[0].run(samples, blocks, columns, 0, a.cols)
		a.max = rs[0].max
		return a, nil
	}

	// Static partitioning: every column costs the same number of DFTs, so
	// there is nothing for a work queue to balance. Ranges are contiguous
	// because a renderer slides its frame buffer and reshapes its window only
	// when `end` changes, both of which are only cheap along a run of columns.
	per := (a.cols + workers - 1) / workers
	var wg sync.WaitGroup
	for w, r := range rs {
		from, to := w*per, min((w+1)*per, a.cols)
		if from >= to {
			break
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			r.run(samples, blocks, columns, from, to)
		}()
	}
	wg.Wait()
	for _, r := range rs {
		if r.max > a.max {
			a.max = r.max
		}
	}
	return a, nil
}

// scheduler walks SoX's streaming accumulation without touching any audio,
// recording which frames each output column is built from. Its arithmetic is
// the original flow/drain loop (spectrogram.c:497-566) with the sample copies
// and the transform removed.
type scheduler struct {
	dftSize, stepSize, blockSteps, xSize int
	blockNorm                            float64

	read      int
	end       int
	endMin    int
	blockNum  int
	fed       int // samples pushed into the sliding buffer so far
	truncated bool

	blocks  []blockPlan
	columns []columnPlan
}

// schedule walks the state machine for nSamples, appending into blocks and
// columns. They are passed in rather than allocated so a Renderer can hand back
// the previous call's arrays; pass nil for a fresh schedule. The scheduler
// itself is always a fresh value, which is what keeps its half-dozen scalar
// state fields from needing an explicit reset.
func schedule(o analyzerOpts, nSamples int, blocks []blockPlan, columns []columnPlan) ([]blockPlan, []columnPlan) {
	s := &scheduler{
		dftSize: o.dftSize, stepSize: o.stepSize,
		blockSteps: o.blockSteps, xSize: o.xSize,
		blockNorm: o.blockNorm,
		blocks:    blocks, columns: columns,
	}
	s.end = o.dftSize                     // spectrogram.c:429
	s.endMin = 0                          // zeroed in start
	s.read = (o.stepSize - o.dftSize) / 2 // spectrogram.c:444 (negative)
	s.flow(nSamples)
	s.drain()
	return s.blocks, s.columns
}

// flow ports spectrogram.c:497-531 for n further samples. The sliding buffer
// always holds the last dftSize samples pushed, so a block's frame is fixed by
// how many have been pushed when it fires, and only that count is tracked here.
func (s *scheduler) flow(n int) {
	idx := 0
	for !s.truncated {
		if s.read == s.stepSize {
			s.read = 0
		}
		if k := min(n-idx, s.stepSize-s.read); k > 0 {
			idx += k
			s.read += k
			s.end -= k
			s.fed += k
		}
		if s.read != s.stepSize {
			break
		}
		s.processBlock()
	}
}

func (s *scheduler) processBlock() {
	if s.end < s.endMin {
		s.end = s.endMin
	}
	s.blocks = append(s.blocks, blockPlan{start: s.fed - s.dftSize, end: s.end})
	s.blockNum++
	if s.blockNum == s.blockSteps {
		s.doColumn()
	}
}

// doColumn ports spectrogram.c:449-475 (non-truncate variant).
func (s *scheduler) doColumn() {
	if len(s.columns) == s.xSize {
		s.truncated = true
		return
	}
	s.columns = append(s.columns, columnPlan{
		firstBlock: len(s.blocks) - s.blockNum,
		nBlocks:    s.blockNum,
		norm:       s.blockNorm,
	})
	s.blockNum = 0
}

// drain ports spectrogram.c:536-566.
func (s *scheduler) drain() {
	if s.truncated {
		return
	}
	isamp := (s.dftSize - s.stepSize) / 2
	leftOver := (isamp + s.read) % s.stepSize
	if leftOver >= s.stepSize>>1 {
		isamp += s.stepSize - leftOver
	}
	s.end = 0
	s.endMin = -s.dftSize
	s.flow(isamp)
	if !s.truncated && s.blockNum != 0 {
		s.blockNorm *= float64(s.blockSteps) / float64(s.blockNum)
		s.doColumn()
	}
}

// columnRenderer is one goroutine's worth of DSP state: a transform plan, a
// window it reshapes as `end` changes, and the scratch for one column.
type columnRenderer struct {
	a    *analysis
	plan *fft.Plan
	ws   *windowState

	buf []float64 // sliding frame, len dftSize
	mag []float64 // per-column power accumulator, len rows
	pow []float64 // per-block power scratch, len rows; nil when a column is one DFT
	db  []float64 // per-column dB scratch, len rows

	frameStart int
	lastEnd    int
	primed     bool // frameStart and lastEnd are meaningful

	max float64
}

// newColumnRenderer allocates the state whose size depends only on the DFT
// geometry, which a Renderer fixes for its lifetime. Everything that varies per
// image is set by reset, which the caller must run before the first use.
func newColumnRenderer(o analyzerOpts, plan *fft.Plan) *columnRenderer {
	return &columnRenderer{
		plan: plan,
		ws:   newWindowState(o.dftSize, o.window),
		buf:  make([]float64, o.dftSize),
		mag:  make([]float64, o.rows),
		db:   make([]float64, o.rows),
	}
}

// reset points a renderer at the image about to be computed and returns its
// carried state to what a freshly built one has.
//
// Both fields matter, and both fail silently rather than loudly if missed.
// Leaving primed set makes loadFrame believe the stream is continuous, so the
// first frame of the new clip slides in over the previous clip's tail instead
// of being read afresh, and it can skip the makeWindow call when the new `end`
// happens to equal the stale one. max is the -n autogain reference, so a peak
// left over from a louder clip renders this one dark.
//
// The remaining buffers need no clearing: loadFrame, PowerInto and Log10 each
// overwrite the whole of the slice they write, and makeWindow rewrites the
// window in full.
func (r *columnRenderer) reset(a *analysis) {
	o := a.opts
	r.a = a
	r.primed = false
	r.frameStart = 0
	r.lastEnd = 0
	r.max = -float64(o.dBRange) // spectrogram.c:443
	// Only a column spanning several DFTs needs somewhere to accumulate from; a
	// single-DFT column transforms straight into mag. blockSteps depends on the
	// clip's sample rate as well as on Options, so a renderer reused across
	// rates can need the accumulator on a later call having not needed it here.
	if o.blockSteps != 1 && r.pow == nil {
		r.pow = make([]float64, o.rows)
	}
}

// run computes columns [from, to) into the analysis output.
func (r *columnRenderer) run(samples []float32, blocks []blockPlan, columns []columnPlan, from, to int) {
	for c := from; c < to; c++ {
		cp := columns[c]
		for i := range cp.nBlocks {
			b := blocks[cp.firstBlock+i]
			r.loadFrame(samples, b.start)
			if !r.primed || b.end != r.lastEnd {
				makeWindow(r.ws, b.end)
				r.lastEnd = b.end
			}
			r.primed = true
			// The first block of a column overwrites the accumulator instead of
			// clearing it and adding, which is the whole of the work at the
			// common one-DFT-per-column geometry. Summing one term is that term.
			if i == 0 {
				r.plan.PowerInto(r.mag, r.buf, r.ws.window)
			} else {
				r.plan.PowerInto(r.pow, r.buf, r.ws.window)
				f64.Add(r.mag, r.mag, r.pow)
			}
		}
		r.emit(c, cp.norm)
	}
}

// loadFrame fills buf with the dftSize samples of the padded stream starting at
// start. Successive frames advance by exactly stepSize, so the common case
// slides the buffer and reads only the new tail; a worker pays the full read
// once, for the first column of its range.
func (r *columnRenderer) loadFrame(samples []float32, start int) {
	dft, step := r.a.opts.dftSize, r.a.opts.stepSize
	if r.primed && start == r.frameStart+step {
		copy(r.buf[:dft-step], r.buf[step:dft])
		fillFrom(r.buf[dft-step:], samples, start+dft-step)
	} else {
		fillFrom(r.buf, samples, start)
	}
	r.frameStart = start
}

// fillFrom writes the padded stream over [from, from+len(dst)) into dst as
// float64. Positions before the first sample or after the last read as zero.
func fillFrom(dst []float64, samples []float32, from int) {
	lo, hi := 0, len(dst)
	if from < 0 {
		lo = min(-from, hi)
	}
	if e := len(samples) - from; e < hi {
		hi = max(e, lo)
	}
	clear(dst[:lo])
	clear(dst[hi:])
	// The drain's last frames sit wholly inside the trailing pad, so there is no
	// overlap with the input at all. Returning here is not just an optimisation:
	// `from` is past the end of samples in that case, and even a zero-length
	// slice taken at an out-of-range index panics.
	if lo >= hi {
		return
	}
	// Sliced to equal lengths so the store is provably in range; this loop sees
	// every sample of the clip once per overlap factor.
	src := samples[from+lo : from+hi]
	d := dst[lo:hi]
	for i, v := range src {
		d[i] = float64(v)
	}
}

// emit converts the accumulated power for column c into the analysis output.
func (r *columnRenderer) emit(c int, norm float64) {
	o := r.a.opts
	// dBfs = 10*log10(mag*norm), vectorized over the whole column: a per-cell
	// math.Log10 would be over half a million calls for a default-sized image.
	//
	// norm is exactly 1 whenever a column is one DFT, and x*1.0 == x for every
	// float64, so that scaling pass is skipped rather than run as a copy.
	src := r.mag
	if norm != 1 {
		f64.Scale(r.db, r.mag, norm)
		src = r.db
	}
	f64.Log10(r.db, src)
	// The factor of 10 rides along in the per-cell loops below rather than in a
	// pass of its own: multiplication commutes exactly, so 10*db[i] is the same
	// float64 that scaling the whole column would have produced, and the column
	// is read once instead of twice.
	gain := float64(o.gain)
	db := r.db[:o.rows]
	base := c * o.rows
	if o.normalize {
		dst := r.a.dBfs[base : base+o.rows]
		for i, v := range db {
			dBfs := 10*v + gain
			dst[i] = float32(dBfs)
			if d := 10 * v; d > r.max {
				r.max = d
			}
		}
		return
	}
	dst := r.a.idx[base : base+o.rows]
	dbRange := float64(o.dBRange)
	for i, v := range db {
		// The round trip through float32 is deliberate, not an accident of the
		// buffer type: the -n path stores float32 and quantises from that, so
		// dropping the precision here too is what keeps the two paths producing
		// the same image, and SoX's own float cache is what set the behaviour.
		dst[i] = uint8(colourIndexAt(float64(float32(10*v+gain)), o.spectrumPoints, dbRange))
	}
}

// quantise resolves the stored dBFS values to palette indices, which the -n
// normalize path can only do once the whole image has been seen and autogain
// is known. It leaves idx in the same column-major layout the default path
// produces, so the raster pass in Render has a single form to handle.
func (a *analysis) quantise(autogain float64) {
	n := a.cols * a.opts.rows
	dbRange := float64(a.opts.dBRange)
	src := a.dBfs[:n]
	dst := a.idx[:n]
	for i, dBfs := range src {
		dst[i] = uint8(colourIndexAt(float64(dBfs)+autogain, a.opts.spectrumPoints, dbRange))
	}
}
