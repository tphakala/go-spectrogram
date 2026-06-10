package mel

import "math"

// fftPlan is a reusable, zero-allocation iterative radix-2 Cooley-Tukey FFT
// over complex64. Twiddle factors and the bit-reversal permutation are
// precomputed once; forward() reuses an internal scratch buffer.
//
// NOTE: this is the hand-rolled FFT that the simd library does not yet provide.
// It is the hot loop of the whole spectrogram. Everything else in this prototype
// (windowing, power, mel projection, stats) already runs on simd kernels. A
// fused real-FFT kernel in simd would replace this file and is where the next
// big speedup lives. See FFT_PRIMITIVE_REQUEST.md.
type fftPlan struct {
	n      int
	bitrev []int
	tw     []complex64 // twiddles e^{-2*pi*i*k/n}, k = 0 .. n/2-1
	buf    []complex64 // scratch, reused every call
}

func isPow2(n int) bool { return n > 0 && n&(n-1) == 0 }

func newFFTPlan(n int) *fftPlan {
	if !isPow2(n) {
		panic("fft: size must be a power of two")
	}
	bits := 0
	for (1 << bits) < n {
		bits++
	}
	br := make([]int, n)
	for i := range br {
		x, r := i, 0
		for b := 0; b < bits; b++ {
			r = (r << 1) | (x & 1)
			x >>= 1
		}
		br[i] = r
	}
	tw := make([]complex64, n/2)
	for k := range tw {
		ang := -2 * math.Pi * float64(k) / float64(n)
		tw[k] = complex64(complex(math.Cos(ang), math.Sin(ang)))
	}
	return &fftPlan{n: n, bitrev: br, tw: tw, buf: make([]complex64, n)}
}

// forward computes the DFT of in (length n) and returns the plan's internal
// buffer holding the result. The result is valid until the next forward() call.
func (p *fftPlan) forward(in []complex64) []complex64 {
	n := p.n
	a := p.buf
	for i := 0; i < n; i++ {
		a[p.bitrev[i]] = in[i]
	}
	// Iterative radix-2 DIT. The innermost butterfly is exactly what a fused
	// simd kernel (c64.ButterflyComplex already exists as a building block)
	// would vectorize across.
	for size := 2; size <= n; size <<= 1 {
		half := size >> 1
		step := n / size
		for i := 0; i < n; i += size {
			k := 0
			for j := i; j < i+half; j++ {
				w := p.tw[k]
				t := w * a[j+half]
				u := a[j]
				a[j] = u + t
				a[j+half] = u - t
				k += step
			}
		}
	}
	return a
}
