package mel

import (
	"math/rand"
	"testing"

	"github.com/tphakala/simd/c64"
	"github.com/tphakala/simd/f32"
)

// Plain-Go scalar equivalents of the simd ops the generator uses. Go does not
// auto-vectorize these, so they stay genuinely scalar, isolating simd's benefit.

func scMul(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func scAbsSq(dst []float32, a []complex64) {
	for i := range dst {
		re, im := real(a[i]), imag(a[i])
		dst[i] = re*re + im*im
	}
}

func scDot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// melPowerScalar is MelPower with every simd call swapped for a scalar loop.
// The FFT (plan.forward) is identical to the simd path.
func (g *Generator) melPowerScalar(signal, dst []float32) int {
	nFrames := NumFrames(len(signal))
	for fr := 0; fr < nFrames; fr++ {
		off := fr * HopLength
		scMul(g.frame, signal[off:off+NFFT], g.window)
		for i := range g.cin {
			g.cin[i] = complex(g.frame[i], 0)
		}
		spec := g.plan.Forward(g.cin)
		scAbsSq(g.power, spec[:NFreq])
		row := dst[fr*NMels : (fr+1)*NMels]
		for i := 0; i < NMels; i++ {
			row[i] = scDot(g.power, g.fb[i])
		}
	}
	return nFrames
}

func benchSig() []float32 {
	sig := make([]float32, SampleRate)
	rng := rand.New(rand.NewSource(11))
	for i := range sig {
		sig[i] = float32(rng.NormFloat64() * 0.1)
	}
	return sig
}

// --- whole pipeline (FFT included, scalar in both): bounds the real benefit ---

func BenchmarkPipelineSIMD(b *testing.B) {
	g := NewGenerator()
	sig := benchSig()
	dst := make([]float32, NumFrames(len(sig))*NMels)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.MelPower(sig, dst)
	}
}

func BenchmarkPipelineScalar(b *testing.B) {
	g := NewGenerator()
	sig := benchSig()
	dst := make([]float32, NumFrames(len(sig))*NMels)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.melPowerScalar(sig, dst)
	}
}

// --- mel projection stage in isolation: where simd should shine most ---

func BenchmarkMelProjSIMD(b *testing.B) {
	g := NewGenerator()
	nf := NumFrames(SampleRate)
	for i := range g.power {
		g.power[i] = float32(i)
	}
	row := make([]float32, NMels)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for fr := 0; fr < nf; fr++ {
			for m := 0; m < NMels; m++ {
				row[m] = f32.DotProduct(g.power, g.fb[m])
			}
		}
	}
}

func BenchmarkMelProjScalar(b *testing.B) {
	g := NewGenerator()
	nf := NumFrames(SampleRate)
	for i := range g.power {
		g.power[i] = float32(i)
	}
	row := make([]float32, NMels)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for fr := 0; fr < nf; fr++ {
			for m := 0; m < NMels; m++ {
				row[m] = scDot(g.power, g.fb[m])
			}
		}
	}
}

// --- window + power stage in isolation ---

func BenchmarkWinPowSIMD(b *testing.B) {
	g := NewGenerator()
	sig := benchSig()
	nf := NumFrames(len(sig))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for fr := 0; fr < nf; fr++ {
			off := fr * HopLength
			f32.Mul(g.frame, sig[off:off+NFFT], g.window)
			c64.FromReal(g.cin, g.frame)
			spec := g.plan.Forward(g.cin)
			c64.AbsSq(g.power, spec[:NFreq])
		}
	}
}

func BenchmarkWinPowScalar(b *testing.B) {
	g := NewGenerator()
	sig := benchSig()
	nf := NumFrames(len(sig))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for fr := 0; fr < nf; fr++ {
			off := fr * HopLength
			scMul(g.frame, sig[off:off+NFFT], g.window)
			for j := range g.cin {
				g.cin[j] = complex(g.frame[j], 0)
			}
			spec := g.plan.Forward(g.cin)
			scAbsSq(g.power, spec[:NFreq])
		}
	}
}
