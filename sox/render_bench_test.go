package sox

import (
	"math"
	"path/filepath"
	"testing"
)

// The four presets birdnet-go renders (see its validSizes map): width in pixels
// with YSize = 2^n+1, so dft = 2*(YSize-1) is a power of two.
var benchPresets = []struct {
	name  string
	xSize int
	ySize int
}{
	{"sm_258x129_dft256", 258, 129},
	{"md_514x257_dft512", 514, 257},
	{"lg_1026x513_dft1024", 1026, 513},
	{"xl_2050x1025_dft2048", 2050, 1025},
}

// benchClip is a 15 s mono clip at 24 kHz, matching birdnet-go's bird profile
// (clips are resampled with `rate 24000` before the spectrogram effect).
const (
	benchSeconds = 15
	benchRate    = 24000
)

func benchSignal(n int, rate float64) []float32 {
	s := make([]float32, n)
	for i := range s {
		t := float64(i) / rate
		s[i] = float32(0.4*math.Sin(2*math.Pi*1000*t) +
			0.2*math.Sin(2*math.Pi*3500*t) +
			0.1*math.Sin(2*math.Pi*7000*t))
	}
	return s
}

// BenchmarkRender covers the full in-process render at birdnet-go's settings,
// with and without chrome. Note that Render stops at an in-memory image, so the
// like-for-like comparison against the sox binary is BenchmarkWritePNG below,
// not this one. See the README for the measured figures; do not restate them
// here, so the two cannot drift apart.
func BenchmarkRender(b *testing.B) {
	sig := benchSignal(benchSeconds*benchRate, benchRate)
	for _, p := range benchPresets {
		for _, raw := range []bool{false, true} {
			name := p.name
			if raw {
				name += "_raw"
			}
			b.Run(name, func(b *testing.B) {
				opt := Options{XSize: p.xSize, YSize: p.ySize, Raw: raw}
				if _, err := Render(sig, benchRate, opt); err != nil {
					b.Fatalf("render: %v", err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					if _, err := Render(sig, benchRate, opt); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkAnalyzer isolates the DSP half (window + DFT + dB conversion) from
// the rasterization half, so a change to one is not masked by the other.
func BenchmarkAnalyzer(b *testing.B) {
	sig := benchSignal(benchSeconds*benchRate, benchRate)
	for _, p := range benchPresets {
		b.Run(p.name, func(b *testing.B) {
			o := normalize(Options{XSize: p.xSize, YSize: p.ySize})
			if err := validate(o); err != nil {
				b.Fatal(err)
			}
			dft, rows := deriveDFTSize(o)
			duration := float64(len(sig)) / benchRate
			xSize, pps := resolveTimeAxis(o, duration)
			actual := makeWindow(newWindowState(dft, o.Window), 0)
			step, blocks, norm := stepSizing(actual, dft, benchRate, pps, o.SlackOverlap)
			opts := analyzerOpts{
				dftSize: dft, rows: rows,
				stepSize: step, blockSteps: blocks, blockNorm: norm,
				gain: -o.Gain, dBRange: o.DBRange,
				spectrumPoints: spectrumPoints(o),
				xSize:          xSize,
				normalize:      o.Normalize,
				window:         o.Window,
				workers:        1,
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := analyze(opts, sig, &analysisScratch{}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkWritePNG is the closest comparison against the SoX binary: Render
// alone produces an in-memory image, whereas invoking `sox` also pays for
// deflate-encoding the PNG and writing it out.
//
// It is still not exact: b.TempDir() is usually tmpfs, and WritePNG does not
// fsync, so the write half is a memcpy into the page cache plus a rename rather
// than real device I/O. Treat it as a CPU comparison.
func BenchmarkWritePNG(b *testing.B) {
	sig := benchSignal(benchSeconds*benchRate, benchRate)
	dir := b.TempDir()
	for _, p := range benchPresets {
		b.Run(p.name, func(b *testing.B) {
			opt := Options{XSize: p.xSize, YSize: p.ySize}
			out := filepath.Join(dir, p.name+".png")
			if err := WritePNG(out, sig, benchRate, opt); err != nil {
				b.Fatalf("write: %v", err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if err := WritePNG(out, sig, benchRate, opt); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkReusedRenderer is BenchmarkRender through a Renderer kept across
// calls, which is how a consumer rendering many clips at one preset uses this
// package. Compare it against BenchmarkRender at the same size: the difference
// is what rebuilding the buffers and the transform plan costs per image.
//
// The Renderer is warmed before the timer starts, so what is measured is the
// steady state rather than the first call, which still allocates everything.
func BenchmarkReusedRenderer(b *testing.B) {
	sig := benchSignal(benchSeconds*benchRate, benchRate)
	for _, p := range benchPresets {
		for _, raw := range []bool{false, true} {
			name := p.name
			if raw {
				name += "_raw"
			}
			b.Run(name, func(b *testing.B) {
				r, err := NewRenderer(Options{XSize: p.xSize, YSize: p.ySize, Raw: raw})
				if err != nil {
					b.Fatalf("new renderer: %v", err)
				}
				if _, err := r.Render(sig, benchRate); err != nil {
					b.Fatalf("render: %v", err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					if _, err := r.Render(sig, benchRate); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkReusedRendererWritePNG is the batch-consumer path end to end, and
// the one where reuse is unambiguously safe: the image never escapes.
func BenchmarkReusedRendererWritePNG(b *testing.B) {
	sig := benchSignal(benchSeconds*benchRate, benchRate)
	dir := b.TempDir()
	for _, p := range benchPresets {
		b.Run(p.name, func(b *testing.B) {
			r, err := NewRenderer(Options{XSize: p.xSize, YSize: p.ySize})
			if err != nil {
				b.Fatalf("new renderer: %v", err)
			}
			out := filepath.Join(dir, "reused_"+p.name+".png")
			if err := r.WritePNG(out, sig, benchRate); err != nil {
				b.Fatalf("write: %v", err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if err := r.WritePNG(out, sig, benchRate); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkRenderSerial is BenchmarkRender pinned to one goroutine, so the
// per-core cost can be tracked separately from what parallelism buys.
func BenchmarkRenderSerial(b *testing.B) {
	sig := benchSignal(benchSeconds*benchRate, benchRate)
	for _, p := range benchPresets {
		b.Run(p.name, func(b *testing.B) {
			opt := Options{XSize: p.xSize, YSize: p.ySize, Workers: 1}
			if _, err := Render(sig, benchRate, opt); err != nil {
				b.Fatalf("render: %v", err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := Render(sig, benchRate, opt); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
