// Command bench times the BSG-BAT mel spectrogram generator on synthetic
// 384 kHz audio and prints a human-readable realtime factor.
package main

import (
	"fmt"
	"math/rand"
	"time"

	mel "github.com/tphakala/go-spectrogram"

	"github.com/tphakala/simd/cpu"
)

func main() {
	fmt.Println("SIMD path:", cpu.Info())

	g := mel.NewGenerator()
	const secs = 10
	sig := make([]float32, mel.SampleRate*secs)
	rng := rand.New(rand.NewSource(42))
	for i := range sig {
		sig[i] = float32(rng.NormFloat64() * 0.1)
	}
	dst := make([]float32, mel.NumFrames(len(sig))*mel.NMels)

	// warmup
	g.MelPower(sig, dst)

	const iters = 20
	start := time.Now()
	for i := 0; i < iters; i++ {
		g.MelPower(sig, dst)
	}
	elapsed := time.Since(start) / iters

	frames := mel.NumFrames(len(sig))
	perAudioSec := elapsed.Seconds() / float64(secs)
	fmt.Printf("audio:            %d s @ %d Hz (%d frames)\n", secs, mel.SampleRate, frames)
	fmt.Printf("mel spectrogram:  %v for %d s of audio\n", elapsed, secs)
	fmt.Printf("per audio-second: %.3f ms\n", perAudioSec*1000)
	fmt.Printf("realtime factor:  %.0fx\n", 1.0/perAudioSec)
	fmt.Printf("CPU budget:       %.2f%% of one core for continuous realtime\n", perAudioSec*100)
}
