package fft

import (
	"fmt"
	"testing"

	"github.com/tphakala/simd/f64"
)

// benchSizes are the transform sizes the sox renderer uses.
var benchSizes = []int{256, 512, 1024, 2048}

// BenchmarkPowerInto measures the vendored transform against simd's
// f64.STFTPlan, which is what it replaces. Run pinned on hybrid-core hosts:
//
//	taskset -c 0,2,4,6 env GOMAXPROCS=4 go test -bench=. -count=10 ./internal/fft/
func BenchmarkPowerInto(b *testing.B) {
	for _, nfft := range benchSizes {
		sig := testSignalF(nfft)
		win := hannF(nfft)
		dst := make([]float64, nfft/2+1)

		b.Run(fmt.Sprintf("nfft%d/vendored", nfft), func(b *testing.B) {
			p, err := NewPlan(nfft)
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				p.PowerInto(dst, sig, win)
			}
		})

		b.Run(fmt.Sprintf("nfft%d/simd", nfft), func(b *testing.B) {
			p, err := f64.NewSTFTPlan(nfft)
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				p.STFTPowerInto(dst, sig, win, nfft, f64.NoPad)
			}
		})
	}
}

// Duplicated from fft_test.go's helpers so the benchmark file stands alone if
// the tests are ever split out.
func testSignalF(n int) []float64 { return testSignal(n) }
func hannF(n int) []float64       { return hann(n) }
