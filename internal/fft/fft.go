// Package fft provides the real-input power-spectrum transform used by the sox
// renderer.
//
// It exists because the transform is the renderer's dominant cost and neither
// available implementation was fast enough: the internal/dsp plan this package
// displaces ran a full-size complex FFT over real input, and simd's
// f64.STFTPlan (still the reference these tests compare against) halves that
// but leaves its own butterfly scalar. BenchmarkPowerInto measures the gap;
// run it rather than trusting a figure quoted here, which rots.
//
// simd#192 has since landed f64.ButterflyComplex and f64.RealFFTUnpack, and
// this package uses the first of them: see stage2 and kernelSpanMin for which
// stages go through the kernel and why the small spans do not. The second is
// deliberately NOT used; unravelPower explains what it measured.
//
// What remains before this package could collapse into a thin call into simd
// is filed as simd#198 (a stage-level butterfly, and a real-FFT unpack that
// writes power directly). The API here is therefore still the minimum the
// renderer needs (a single frame, no framing or padding modes) rather than a
// general STFT.
//
// The transform is the standard real-input trick: an nfft-point real sequence
// is packed into an nfft/2-point complex sequence (evens into the real part,
// odds into the imaginary part), transformed at half size, then unravelled back
// into the Hermitian half-spectrum.
package fft

import (
	"errors"
	"fmt"
	"math"
	"math/bits"

	"github.com/tphakala/simd/f64"
)

// ErrUnsupportedSize is returned by NewPlan when nfft is not a power of two, or
// is a power of two below 4. It is deliberately not named for the power-of-two
// condition alone: nfft == 2 is a power of two and is still rejected, because
// the packed transform needs at least two complex points.
var ErrUnsupportedSize = errors.New("fft: size must be a power of two >= 4")

// stage holds one pass of the transform: its radix, the span it operates at,
// and its twiddles laid out contiguously by j rather than read at stride from
// one shared table.
//
// The twiddles a butterfly needs are W^t, W^2t and W^3t at t = j*step, so a
// shared table is walked at stride step, 2*step and 3*step. Those strides are
// what the loop was spending its loads on: the values depend only on j and the
// stage, never on the frame, so they are resolved once at plan time into
// sequential arrays. Total extra memory is a little over 2*half float64 across
// all stages. A radix-2 stage uses only the first pair.
type stage struct {
	radix    int
	span     int
	w1r, w1i []float64
	w2r, w2i []float64
	w3r, w3i []float64
}

// kernelSpanMin is the span from which a stage switches from the scalar radix-4
// butterfly to simd's vectorized radix-2 one.
//
// f64.ButterflyComplex takes one contiguous run of pairs, so a stage at span s
// costs half/(2s) calls, and below a certain span the per-call overhead is
// larger than the butterflies themselves. Measured on an i7-1260P over a
// half of 1024, per stage: the kernel is 9.7x the scalar radix-2 loop at span
// 512 but only 1.3x at span 4, while one scalar radix-4 stage does the work of
// two radix-2 stages for well under twice the cost. Comparing like for like,
// radix 4 wins below span 16 and the kernel wins from there up.
//
// A stage-level kernel (one call per stage, taking the span) would remove the
// trade entirely and is worth asking simd for; it would make the small spans as
// fast per butterfly as the large ones.
const kernelSpanMin = 8

// Plan holds the resident twiddle tables and scratch for a fixed transform
// size. Reuse one across frames to stay allocation-free.
//
// A Plan carries per-transform scratch, so its methods are NOT safe for
// concurrent use on the same Plan; use one per goroutine.
type Plan struct {
	nfft int
	half int // nfft/2: the size of the packed complex transform

	// leadRadix2 records that the decomposition needs a radix-2 pass, which
	// happens exactly when log2(half) is odd. Radix 4 does the same work as two
	// radix-2 stages in one pass over the scratch, with three quarters of the
	// complex multiplies, so the decomposition is all 4s with a single leading 2
	// when it cannot be all 4s.
	//
	// Either way the leading pass runs at span 1, where every twiddle is W^0 = 1
	// and no complex multiply is needed at all. That is why it is not in stages:
	// pack fuses it into the load (see packStage), which both skips the
	// multiplies and saves a whole read-modify-write pass over the scratch.
	leadRadix2 bool

	// stages lists the radix-4 passes after the fused leading one, from the
	// smallest span outwards.
	stages []stage

	// iperm[d] is the source index whose sample belongs at scratch position d,
	// the inverse of the mixed-radix digit reversal that decimation in time
	// expects. It is the inverse rather than the forward map so that pack can
	// gather: the leading butterfly consumes four (or two) consecutive
	// destinations at a time, and writing them sequentially is what lets the
	// leading stage be fused into the load in the first place.
	iperm []int32

	// Unravel twiddles W_N^k = exp(-2*pi*i*k/nfft) for k in [0, half].
	unRe, unIm []float64

	// Scratch holding the packed complex frame, transformed in place.
	re, im []float64
}

// NewPlan builds a reusable plan for nfft-point real-input transforms. nfft
// must be a power of two and at least 4.
func NewPlan(nfft int) (*Plan, error) {
	if nfft < 4 || nfft&(nfft-1) != 0 {
		return nil, ErrUnsupportedSize
	}
	half := nfft >> 1

	p := &Plan{
		nfft:  nfft,
		half:  half,
		iperm: make([]int32, half),
		unRe:  make([]float64, half+1),
		unIm:  make([]float64, half+1),
		re:    make([]float64, half),
		im:    make([]float64, half),
	}

	// half is a power of two, so its trailing-zero count is log2(half).
	logHalf := bits.TrailingZeros(uint(half))
	p.leadRadix2 = logHalf%2 == 1

	radices := decompose(logHalf)
	p.buildIPerm(radices)
	p.buildStages(radices)

	// Twiddles are evaluated in float64 from the exact angle, matching the
	// reference implementation this replaces bit for bit.
	for k := 0; k <= half; k++ {
		s, c := math.Sincos(2 * math.Pi * float64(k) / float64(nfft))
		p.unRe[k], p.unIm[k] = c, -s
	}

	return p, nil
}

// decompose chooses the radix of each pass, from the smallest span outwards,
// for a transform of size 2^logHalf.
//
// Radix 4 does the same work as two radix-2 stages in one pass over the
// scratch, with three quarters of the complex multiplies, so it is the better
// scalar choice and the decomposition starts with it. It only fits an even
// number of remaining halvings, hence the leading radix-2 when logHalf is odd;
// that leading pass runs at span 1 where every twiddle is 1, so it is the
// cheapest place to put it. From kernelSpanMin the passes switch to radix 2,
// because that is the shape simd's vectorized butterfly takes and from that
// span it is worth far more than the multiplies radix 4 saves.
func decompose(logHalf int) []int {
	var radices []int
	span, left := 1, logHalf
	if left%2 == 1 {
		radices = append(radices, 2)
		span, left = 2, left-1
	}
	for left > 0 {
		if left >= 2 && span < kernelSpanMin {
			radices = append(radices, 4)
			span, left = span*4, left-2
			continue
		}
		radices = append(radices, 2)
		span, left = span*2, left-1
	}
	return radices
}

// buildIPerm fills iperm with the inverse of the mixed-radix digit reversal
// that decimation in time expects.
//
// The digit weights run opposite to the order the stages are applied: the stage
// with span 1 corresponds to the LAST factor of the decimation, so the index is
// decomposed against the reversed radix list and then read back with the digits
// in the opposite order. For an all-radix-2 decomposition this reduces to plain
// bit reversal, and when every radix is equal the direction cannot be observed
// at all, which is why only a mixed decomposition such as [2 4 4] pins it down.
//
// Digit reversal is an involution only when all the radices are equal, so the
// inverse is stored explicitly rather than assumed symmetric.
func (p *Plan) buildIPerm(radices []int) {
	n := len(radices)
	rad := make([]int, n)
	for d, r := range radices {
		rad[n-1-d] = r
	}
	digits := make([]int, n)
	for i := range p.iperm {
		v := i
		for d := range n {
			digits[d] = v % rad[d]
			v /= rad[d]
		}
		r := 0
		for d := range n {
			r = r*rad[d] + digits[d]
		}
		p.iperm[r] = int32(i)
	}
}

// buildStages resolves the per-stage twiddle tables for every pass after the
// fused leading one.
func (p *Plan) buildStages(radices []int) {
	// Span after the leading pass: 2 for a radix-2 lead, 4 otherwise.
	span := radices[0]
	// tw returns the shared twiddle W^t = exp(-2*pi*i*t/half), evaluated from
	// the exact angle so the values are identical to a strided read of one
	// table built the same way.
	tw := func(t int) (re, im float64) {
		s, c := math.Sincos(2 * math.Pi * float64(t) / float64(p.half))
		return c, -s
	}
	for _, r := range radices[1:] {
		step := p.half / (span * r)
		st := stage{
			radix: r, span: span,
			w1r: make([]float64, span), w1i: make([]float64, span),
		}
		if r == 4 {
			st.w2r, st.w2i = make([]float64, span), make([]float64, span)
			st.w3r, st.w3i = make([]float64, span), make([]float64, span)
		}
		for j := range span {
			t := j * step
			st.w1r[j], st.w1i[j] = tw(t)
			if r == 4 {
				st.w2r[j], st.w2i[j] = tw(2 * t)
				st.w3r[j], st.w3i[j] = tw(3 * t)
			}
		}
		p.stages = append(p.stages, st)
		span *= r
	}
}

// NFFT returns the transform size the plan was built for.
func (p *Plan) NFFT() int { return p.nfft }

// NumBins returns nfft/2 + 1, the number of bins in the Hermitian
// half-spectrum (DC through Nyquist).
func (p *Plan) NumBins() int { return p.half + 1 }

// PowerInto writes the real-input power spectrum |X_k|^2 of one nfft-sample
// frame into dst[0:NumBins]. window, when non-nil, is applied during the load.
//
// It panics unless len(dst) >= NumBins(), len(signal) >= NFFT(), and, for a
// non-nil window, len(window) >= NFFT(). The check is explicit because the
// loads below re-slice these arguments, and a slice expression bound-checks
// against capacity rather than length: without it, a short slice backed by a
// larger array would be accepted silently and the transform would read whatever
// followed it, producing a plausible but wrong spectrum instead of failing.
func (p *Plan) PowerInto(dst, signal, window []float64) {
	if len(dst) < p.half+1 || len(signal) < p.nfft || (window != nil && len(window) < p.nfft) {
		panic(fmt.Sprintf("fft: PowerInto: nfft %d needs dst >= %d, signal >= %d, window >= %d (nil ok); got %d, %d, %d",
			p.nfft, p.half+1, p.nfft, p.nfft, len(dst), len(signal), len(window)))
	}
	p.packStage(signal, window)
	p.transform()
	p.unravelPower(dst)
}

// packStage loads the frame as half complex samples c[j] = x[2j] + i*x[2j+1],
// applying the window on the way in, permuting into digit-reversed order, and
// applying the leading twiddle-free butterfly, all in one pass.
//
// The load is a gather rather than a scatter: it walks the destination in
// order, pulling each sample from iperm. That is what makes fusing the leading
// stage possible, since one butterfly's outputs are consecutive destinations,
// and it keeps every store sequential.
func (p *Plan) packStage(signal, window []float64) {
	re, im, ip := p.re[:p.half], p.im[:p.half], p.iperm[:p.half]
	src := signal[:2*len(ip)]

	if window == nil {
		if p.leadRadix2 {
			for q := 0; q+2 <= len(re); q += 2 {
				a, b := 2*int(ip[q]), 2*int(ip[q+1])
				ar, ai := src[a], src[a+1]
				br, bi := src[b], src[b+1]
				re[q], im[q] = ar+br, ai+bi
				re[q+1], im[q+1] = ar-br, ai-bi
			}
			return
		}
		for q := 0; q+4 <= len(re); q += 4 {
			a, b := 2*int(ip[q]), 2*int(ip[q+1])
			c, d := 2*int(ip[q+2]), 2*int(ip[q+3])
			ar, ai := src[a], src[a+1]
			br, bi := src[b], src[b+1]
			cr, ci := src[c], src[c+1]
			dr, di := src[d], src[d+1]
			t0r, t0i := ar+cr, ai+ci
			t1r, t1i := ar-cr, ai-ci
			t2r, t2i := br+dr, bi+di
			t3r, t3i := br-dr, bi-di
			re[q], im[q] = t0r+t2r, t0i+t2i
			re[q+2], im[q+2] = t0r-t2r, t0i-t2i
			re[q+1], im[q+1] = t1r+t3i, t1i-t3r
			re[q+3], im[q+3] = t1r-t3i, t1i+t3r
		}
		return
	}

	w := window[:2*len(ip)]
	if p.leadRadix2 {
		for q := 0; q+2 <= len(re); q += 2 {
			a, b := 2*int(ip[q]), 2*int(ip[q+1])
			ar, ai := src[a]*w[a], src[a+1]*w[a+1]
			br, bi := src[b]*w[b], src[b+1]*w[b+1]
			re[q], im[q] = ar+br, ai+bi
			re[q+1], im[q+1] = ar-br, ai-bi
		}
		return
	}
	for q := 0; q+4 <= len(re); q += 4 {
		a, b := 2*int(ip[q]), 2*int(ip[q+1])
		c, d := 2*int(ip[q+2]), 2*int(ip[q+3])
		ar, ai := src[a]*w[a], src[a+1]*w[a+1]
		br, bi := src[b]*w[b], src[b+1]*w[b+1]
		cr, ci := src[c]*w[c], src[c+1]*w[c+1]
		dr, di := src[d]*w[d], src[d+1]*w[d+1]
		t0r, t0i := ar+cr, ai+ci
		t1r, t1i := ar-cr, ai-ci
		t2r, t2i := br+dr, bi+di
		t3r, t3i := br-dr, bi-di
		re[q], im[q] = t0r+t2r, t0i+t2i
		re[q+2], im[q+2] = t0r-t2r, t0i-t2i
		re[q+1], im[q+1] = t1r+t3i, t1i-t3r
		re[q+3], im[q+3] = t1r-t3i, t1i+t3r
	}
}

// transform runs the radix-4 stages that packStage did not already apply, in
// place over the scratch.
func (p *Plan) transform() {
	for i := range p.stages {
		if s := &p.stages[i]; s.radix == 2 {
			p.stage2(s)
		} else {
			p.stage4(s)
		}
	}
}

// stage2 applies one radix-2 decimation-in-time stage through simd's
// vectorized butterfly, which is exactly this operation over a contiguous run
// of pairs: temp = lower*W, then lower = upper-temp and upper = upper+temp.
//
// One call per block is what the kernel's shape allows, so this is only used
// from kernelSpanMin up, where the runs are long enough for that to pay.
func (p *Plan) stage2(s *stage) {
	re, im := p.re[:p.half], p.im[:p.half]
	span := s.span
	m := span * 2
	for k := 0; k+m <= len(re); k += m {
		f64.ButterflyComplex(
			re[k:k+span:k+span], im[k:k+span:k+span],
			re[k+span:k+m:k+m], im[k+span:k+m:k+m],
			s.w1r, s.w1i)
	}
}

// stage4 applies one radix-4 decimation-in-time stage. With A, B, C, D the four
// twiddled inputs, the forward butterfly (W = exp(-2*pi*i/N)) is
//
//	X0 = (A+C) + (B+D)
//	X1 = (A-C) - i*(B-D)
//	X2 = (A+C) - (B+D)
//	X3 = (A-C) + i*(B-D)
//
// which is why the two cross terms need no multiply: rotating by -i is a swap
// of real and imaginary parts with one sign flip.
func (p *Plan) stage4(s *stage) {
	re, im := p.re[:p.half], p.im[:p.half]
	span := s.span
	m := span * 4
	// Slicing every twiddle array to exactly span is what proves w1r[j] and its
	// siblings in range for the inner loop, which ranges over a slice of that
	// same length.
	w1r, w1i := s.w1r[:span], s.w1i[:span]
	w2r, w2i := s.w2r[:span], s.w2i[:span]
	w3r, w3i := s.w3r[:span], s.w3i[:span]
	for k := 0; k+m <= len(re); k += m {
		r0 := re[k : k+span : k+span]
		r1 := re[k+span : k+2*span : k+2*span]
		r2 := re[k+2*span : k+3*span : k+3*span]
		r3 := re[k+3*span : k+4*span : k+4*span]
		m0 := im[k : k+span : k+span]
		m1 := im[k+span : k+2*span : k+2*span]
		m2 := im[k+2*span : k+3*span : k+3*span]
		m3 := im[k+3*span : k+4*span : k+4*span]
		for j := range r0 {
			ar, ai := r0[j], m0[j]

			br := w1r[j]*r1[j] - w1i[j]*m1[j]
			bi := w1r[j]*m1[j] + w1i[j]*r1[j]
			cr := w2r[j]*r2[j] - w2i[j]*m2[j]
			ci := w2r[j]*m2[j] + w2i[j]*r2[j]
			dr := w3r[j]*r3[j] - w3i[j]*m3[j]
			di := w3r[j]*m3[j] + w3i[j]*r3[j]

			t0r, t0i := ar+cr, ai+ci
			t1r, t1i := ar-cr, ai-ci
			t2r, t2i := br+dr, bi+di
			t3r, t3i := br-dr, bi-di

			r0[j], m0[j] = t0r+t2r, t0i+t2i
			r2[j], m2[j] = t0r-t2r, t0i-t2i
			r1[j], m1[j] = t1r+t3i, t1i-t3r
			r3[j], m3[j] = t1r-t3i, t1i+t3r
		}
	}
}

// unravelPower recombines the half-size spectrum into |X_k|^2 for k in
// [0, half].
//
// With C the half-size spectrum, the even and odd half-spectra of the original
// real sequence are E = (C[k] + conj(C[half-k]))/2 and
// O = -i*(C[k] - conj(C[half-k]))/2, and X[k] = E + W_N^k * O.
// Bins k and half-k read the same pair of half-spectrum values, just swapped,
// so they are emitted together from one set of loads. Writing E and O for the
// terms derived from that pair, bin half-k reuses them with two sign flips:
//
//	X[k]      = ( E.r + w.r*O.r - w.i*O.i,   E.i + w.r*O.i + w.i*O.r )
//	X[half-k] = ( E.r + v.r*O.r + v.i*O.i,  -E.i - v.r*O.i + v.i*O.r )
//
// with w the unravel twiddle at k and v the one at half-k.
//
// f64.RealFFTUnpack (simd v1.6.0) is the vectorized form of this
// recombination, and it loses to this loop: measured on an idle i7-1260P at a
// fixed clock, +39% at nfft 256, +15% at 512, level at 1024 and +4% at 2048.
// It writes the complex half-spectrum, so squaring it into power costs two
// further passes over the bins, and that traffic is worth more here than the
// vector lanes. A kernel that wrote |X_k|^2 directly would flip the result;
// that is the shape to ask simd for.
func (p *Plan) unravelPower(dst []float64) {
	half := p.half
	re, im := p.re[:half], p.im[:half]
	unRe, unIm := p.unRe[:half+1], p.unIm[:half+1]
	dst = dst[:half+1]

	// DC and Nyquist both fold onto C[0] and are purely real.
	c0r, c0i := re[0], im[0]
	dst[0] = (c0r + c0i) * (c0r + c0i)
	dst[half] = (c0r - c0i) * (c0r - c0i)

	for k := 1; k < half-k; k++ {
		m := half - k
		akr, aki := re[k], im[k]
		bkr, bki := re[m], im[m]

		er := 0.5 * (akr + bkr)
		ei := 0.5 * (aki - bki)
		or := 0.5 * (aki + bki)
		oi := -0.5 * (akr - bkr)

		wr, wi := unRe[k], unIm[k]
		xr := er + (wr*or - wi*oi)
		xi := ei + (wr*oi + wi*or)
		dst[k] = xr*xr + xi*xi

		vr, vi := unRe[m], unIm[m]
		yr := er + (vr*or + vi*oi)
		yi := -ei - (vr*oi - vi*or)
		dst[m] = yr*yr + yi*yi
	}

	// The self-paired middle bin, present whenever half is even.
	if h := half / 2; h*2 == half && h > 0 {
		akr, aki := re[h], im[h]
		er, ei := akr, 0.0
		or, oi := aki, 0.0
		wr, wi := unRe[h], unIm[h]
		xr := er + (wr*or - wi*oi)
		xi := ei + (wr*oi + wi*or)
		dst[h] = xr*xr + xi*xi
	}
}

// Clone returns a plan for the same size that shares this one's twiddle tables
// and permutation but carries its own scratch, so the two can transform
// concurrently.
//
// The shared state is written once in NewPlan and only read afterwards. Cloning
// exists because building a plan costs a few thousand Sincos calls, which is a
// real fraction of a single spectrogram; a clone costs two allocations.
func (p *Plan) Clone() *Plan {
	c := *p
	c.re = make([]float64, p.half)
	c.im = make([]float64, p.half)
	return &c
}
