// Package fft provides the real-input power-spectrum transform used by the sox
// renderer.
//
// It exists because simd's f64.STFTPlan, which this replaces, runs a scalar
// radix-2 butterfly: on arm64 it accounts for around 61% of a spectrogram
// render. See github.com/tphakala/simd#192. Once that lands vectorized f64
// kernels this package should shrink back to a thin call into simd, so the API
// here is deliberately the minimum the renderer needs (a single frame, no
// framing or padding modes) rather than a general STFT.
//
// The transform is the standard real-input trick: an nfft-point real sequence
// is packed into an nfft/2-point complex sequence (evens into the real part,
// odds into the imaginary part), transformed at half size, then unravelled back
// into the Hermitian half-spectrum.
package fft

import (
	"errors"
	"math"
)

// ErrNotPowerOfTwo is returned when nfft is not a power of two >= 4.
var ErrNotPowerOfTwo = errors.New("fft: size must be a power of two >= 4")

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
		return nil, ErrNotPowerOfTwo
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

	logHalf := 0
	for 1<<logHalf < half {
		logHalf++
	}
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

// NumBins returns nfft/2 + 1, the number of bins in the Hermitian
// half-spectrum (DC through Nyquist).
func (p *Plan) NumBins() int { return p.half + 1 }

// PowerInto writes the real-input power spectrum |X_k|^2 of one nfft-sample
// frame into dst[0:NumBins]. window, when non-nil, is applied during the load
// and must be at least nfft long. signal must be at least nfft long.
func (p *Plan) PowerInto(dst, signal, window []float64) {
	p.pack(signal, window)
	p.transform()
	p.unravelPower(dst)
}

// pack loads the frame as half complex samples c[j] = x[2j] + i*x[2j+1],
// applying the window on the way in, and applies the bit-reversal permutation
// so the transform can run in place.
func (p *Plan) pack(signal, window []float64) {
	re, im, br := p.re, p.im, p.perm
	src := signal[:2*p.half]
	if window == nil {
		for j, r := range br {
			re[r] = src[2*j]
			im[r] = src[2*j+1]
		}
		return
	}
	w := window[:2*p.half]
	for j, r := range br {
		re[r] = src[2*j] * w[2*j]
		im[r] = src[2*j+1] * w[2*j+1]
	}
}

// transform runs an in-place decimation-in-time complex FFT of size half over
// the scratch, which pack has already left in digit-reversed order.
func (p *Plan) transform() {
	span := 1
	for _, r := range p.radices {
		if r == 2 {
			// Only ever the first stage; see stage2.
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
	re, im := p.re, p.im
	m := span * 4
	step := p.half / m
	for k := 0; k < p.half; k += m {
		for j := range span {
			i0 := k + j
			i1 := i0 + span
			i2 := i1 + span
			i3 := i2 + span

			// A is untwiddled; B, C, D carry W^j, W^2j, W^3j.
			ar, ai := re[i0], im[i0]

			t := j * step
			w1r, w1i := p.twRe[t], p.twIm[t]
			w2r, w2i := p.twRe[2*t], p.twIm[2*t]
			w3r, w3i := p.twRe[3*t], p.twIm[3*t]

			br := w1r*re[i1] - w1i*im[i1]
			bi := w1r*im[i1] + w1i*re[i1]
			cr := w2r*re[i2] - w2i*im[i2]
			ci := w2r*im[i2] + w2i*re[i2]
			dr := w3r*re[i3] - w3i*im[i3]
			di := w3r*im[i3] + w3i*re[i3]

			// t0 = A+C, t1 = A-C, t2 = B+D, t3 = B-D.
			t0r, t0i := ar+cr, ai+ci
			t1r, t1i := ar-cr, ai-ci
			t2r, t2i := br+dr, bi+di
			t3r, t3i := br-dr, bi-di

			re[i0], im[i0] = t0r+t2r, t0i+t2i
			re[i2], im[i2] = t0r-t2r, t0i-t2i
			// -i*(t3r + i*t3i) = t3i - i*t3r
			re[i1], im[i1] = t1r+t3i, t1i-t3r
			re[i3], im[i3] = t1r-t3i, t1i+t3r
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
	re, im := p.re, p.im
	half := p.half

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

		wr, wi := p.unRe[k], p.unIm[k]
		xr := er + (wr*or - wi*oi)
		xi := ei + (wr*oi + wi*or)
		dst[k] = xr*xr + xi*xi

		vr, vi := p.unRe[m], p.unIm[m]
		yr := er + (vr*or + vi*oi)
		yi := -ei - (vr*oi - vi*or)
		dst[m] = yr*yr + yi*yi
	}

	// The self-paired middle bin, present whenever half is even.
	if h := half / 2; h*2 == half && h > 0 {
		akr, aki := re[h], im[h]
		er, ei := akr, 0.0
		or, oi := aki, 0.0
		wr, wi := p.unRe[h], p.unIm[h]
		xr := er + (wr*or - wi*oi)
		xi := ei + (wr*oi + wi*or)
		dst[h] = xr*xr + xi*xi
	}
}
