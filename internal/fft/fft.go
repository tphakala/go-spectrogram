// Package fft provides the real-input power-spectrum transform used by the sox
// renderer.
//
// It exists because the transform is the renderer's dominant cost and neither
// available implementation was fast enough: the internal/dsp plan this package
// displaces ran a full-size complex FFT over real input, and simd's
// f64.STFTPlan (evaluated at v1.5.0, and the reference these tests still
// compare against) halves that but leaves the butterfly scalar. Measured on an
// i7-1260P at nfft=1024, this package is about 1.6x f64.STFTPlan.
//
// simd#192 tracks vectorizing that butterfly and adding the f64 kernels this
// would need; f64 currently exports no ButterflyComplex or RealFFTUnpack at
// all, so composing simd primitives was not an option for a float64 consumer.
// Once #192 lands, this package should shrink back to a thin call into simd,
// so the API here is deliberately the minimum the renderer needs (a single
// frame, no framing or padding modes) rather than a general STFT.
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
)

// ErrUnsupportedSize is returned by NewPlan when nfft is not a power of two, or
// is a power of two below 4. It is deliberately not named for the power-of-two
// condition alone: nfft == 2 is a power of two and is still rejected, because
// the packed transform needs at least two complex points.
var ErrUnsupportedSize = errors.New("fft: size must be a power of two >= 4")

// Plan holds the resident twiddle tables and scratch for a fixed transform
// size. Reuse one across frames to stay allocation-free.
//
// A Plan carries per-transform scratch, so its methods are NOT safe for
// concurrent use on the same Plan; use one per goroutine.
type Plan struct {
	nfft int
	half int // nfft/2: the size of the packed complex transform

	// radices lists the transform stages in the order they are applied, from
	// the smallest span outwards. Radix 4 does the same work as two radix-2
	// stages in one pass over the scratch, with three quarters of the complex
	// multiplies, so the decomposition is all 4s with a single leading 2 when
	// log2(half) is odd.
	radices []int

	// perm is the mixed-radix digit reversal for radices, applied during the
	// load so the stages can run in place with no separate reorder pass.
	perm []int32

	// Twiddles for the size-half complex FFT, twRe[t] = cos(2*pi*t/half) and
	// twIm[t] = -sin(2*pi*t/half). Radix 4 reaches 3*j*step, so unlike a
	// purely radix-2 table this must span the full [0, half).
	twRe, twIm []float64

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
		nfft: nfft,
		half: half,
		perm: make([]int32, half),
		twRe: make([]float64, half),
		twIm: make([]float64, half),
		unRe: make([]float64, half+1),
		unIm: make([]float64, half+1),
		re:   make([]float64, half),
		im:   make([]float64, half),
	}

	// half is a power of two, so its trailing-zero count is log2(half).
	logHalf := bits.TrailingZeros(uint(half))
	if logHalf%2 == 1 {
		p.radices = append(p.radices, 2)
	}
	for r := logHalf % 2; r < logHalf; r += 2 {
		p.radices = append(p.radices, 4)
	}
	p.buildPerm()

	// Twiddles are evaluated in float64 from the exact angle, matching the
	// reference implementation this replaces bit for bit.
	for t := range p.twRe {
		s, c := math.Sincos(2 * math.Pi * float64(t) / float64(half))
		p.twRe[t], p.twIm[t] = c, -s
	}
	for k := 0; k <= half; k++ {
		s, c := math.Sincos(2 * math.Pi * float64(k) / float64(nfft))
		p.unRe[k], p.unIm[k] = c, -s
	}

	return p, nil
}

// buildPerm fills perm with the mixed-radix digit reversal that decimation in
// time expects.
//
// The digit weights run opposite to the order the stages are applied: the stage
// with span 1 corresponds to the LAST factor of the decimation, so the index is
// decomposed against the reversed radix list and then read back with the digits
// in the opposite order. For an all-radix-2 decomposition this reduces to plain
// bit reversal, and when every radix is equal the direction cannot be observed
// at all, which is why only a mixed decomposition such as [2 4 4] pins it down.
func (p *Plan) buildPerm() {
	n := len(p.radices)
	rad := make([]int, n)
	for d, r := range p.radices {
		rad[n-1-d] = r
	}
	digits := make([]int, n)
	for i := range p.perm {
		v := i
		for d := range n {
			digits[d] = v % rad[d]
			v /= rad[d]
		}
		r := 0
		for d := range n {
			r = r*rad[d] + digits[d]
		}
		p.perm[i] = int32(r)
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
	p.pack(signal, window)
	p.transform()
	p.unravelPower(dst)
}

// pack loads the frame as half complex samples c[j] = x[2j] + i*x[2j+1],
// applying the window on the way in, and applies the bit-reversal permutation
// so the transform can run in place.
func (p *Plan) pack(signal, window []float64) {
	re, im, br := p.re[:p.half], p.im[:p.half], p.perm
	src := signal[:2*len(br)]
	if window == nil {
		for j, r := range br {
			s := src[2*j : 2*j+2 : 2*j+2]
			re[r] = s[0]
			im[r] = s[1]
		}
		return
	}
	w := window[:2*len(br)]
	for j, r := range br {
		s := src[2*j : 2*j+2 : 2*j+2]
		ww := w[2*j : 2*j+2 : 2*j+2]
		re[r] = s[0] * ww[0]
		im[r] = s[1] * ww[1]
	}
}

// transform runs an in-place decimation-in-time complex FFT of size half over
// the scratch, which pack has already left in digit-reversed order.
func (p *Plan) transform() {
	span := 1
	for _, r := range p.radices {
		if r == 2 {
			// stage2 hardcodes span 1 (see its doc); a radix-2 anywhere but
			// first would silently produce a wrong transform, so assert rather
			// than trust the construction in NewPlan.
			if span != 1 {
				panic("fft: radix-2 stage at span > 1")
			}
			p.stage2()
		} else {
			p.stage4(span)
		}
		span *= r
	}
}

// stage2 applies the single radix-2 stage that a decomposition needs when
// log2(half) is odd. It is always the first stage, so its span is 1 and the
// only twiddle it could use is W^0 = 1: the complex multiply is skipped
// entirely and each butterfly is one complex add and one complex subtract.
func (p *Plan) stage2() {
	re, im := p.re, p.im
	for k := 0; k < len(re); k += 2 {
		ar, ai := re[k], im[k]
		br, bi := re[k+1], im[k+1]
		re[k], im[k] = ar+br, ai+bi
		re[k+1], im[k+1] = ar-br, ai-bi
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
func (p *Plan) stage4(span int) {
	re, im := p.re[:p.half], p.im[:p.half]
	m := span * 4
	step := p.half / m
	twRe, twIm := p.twRe[:p.half], p.twIm[:p.half]
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

			t := j * step
			w1r, w1i := twRe[t], twIm[t]
			w2r, w2i := twRe[2*t], twIm[2*t]
			w3r, w3i := twRe[3*t], twIm[3*t]

			br := w1r*r1[j] - w1i*m1[j]
			bi := w1r*m1[j] + w1i*r1[j]
			cr := w2r*r2[j] - w2i*m2[j]
			ci := w2r*m2[j] + w2i*r2[j]
			dr := w3r*r3[j] - w3i*m3[j]
			di := w3r*m3[j] + w3i*r3[j]

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
