package mel

import (
	"math/rand"
	"testing"

	"github.com/tphakala/simd/c64"
	"github.com/tphakala/simd/f32"
)

// These split the per-second cost into its stages so we know what to optimize.
// All run over a full second of 384 kHz audio (NumFrames(SampleRate) frames).

func makeSig() []float32 {
	sig := make([]float32, SampleRate)
	rng := rand.New(rand.NewSource(7))
	for i := range sig {
		sig[i] = float32(rng.NormFloat64() * 0.1)
	}
	return sig
}

// FFT stage only: window + real->complex + FFT + power, no mel projection.
func BenchmarkStageFFTandPower(b *testing.B) {
	g := NewGenerator()
	sig := makeSig()
	nf := NumFrames(len(sig))
	b.ReportAllocs()
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

// FFT forward only (the scalar hot loop a simd kernel would replace).
func BenchmarkStageFFTForwardOnly(b *testing.B) {
	g := NewGenerator()
	sig := makeSig()
	nf := NumFrames(len(sig))
	// pre-window into complex inputs so we time only forward()
	c64.FromReal(g.cin, g.frame)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for fr := 0; fr < nf; fr++ {
			g.plan.Forward(g.cin)
		}
	}
}

// Mel projection stage only: 128 dot products of length 513 per frame (simd).
func BenchmarkStageMelProjection(b *testing.B) {
	g := NewGenerator()
	nf := NumFrames(SampleRate)
	for i := range g.power {
		g.power[i] = float32(i)
	}
	dstRow := make([]float32, NMels)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for fr := 0; fr < nf; fr++ {
			for m := 0; m < NMels; m++ {
				dstRow[m] = f32.DotProduct(g.power, g.fb[m])
			}
		}
	}
}
