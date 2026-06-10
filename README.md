# go-spectrogram

A Go spectrogram library built on [github.com/tphakala/simd](https://github.com/tphakala/simd).

It started as a prototype mel-spectrogram generator for BSG-BAT (the
spectrogram-input bat CNN, [Zenodo 10.5281/zenodo.15495676](https://doi.org/10.5281/zenodo.15495676)),
written to answer "can the spectrogram run in realtime in Go" with a hard
in-language number and to scope what `simd` needs. It now supports multiple
spectrogram types: the original mel-tensor generator (`mel`) and a
SoX-compatible image renderer (`sox`), over a shared DSP core.

## Current result (i7-1260P, AVX+FMA)

- **8.9 ms** to compute the mel spectrogram for **1 second** of 384 kHz audio.
- **~106x realtime**, **0.94%** of one core, **zero allocations** in the hot path.
- The FFT is currently a hand-rolled scalar radix-2 (the one stage not on simd,
  ~63% of runtime). simd issues [#108](https://github.com/tphakala/simd/issues/108)
  (real-input STFT) and [#109](https://github.com/tphakala/simd/issues/109)
  (Log/Log10) track moving it onto the fast path.

```
go test ./...                              # correctness (FFT tone, shapes, normalize range)
go test -bench=. -benchmem -run=XXX        # benchmarks (incl. simd vs scalar)
go run ./cmd/bench                         # friendly realtime number + active SIMD path
```

The module depends on [`github.com/tphakala/simd`](https://github.com/tphakala/simd)
(pinned to `v1.2.0-rc.5` in `go.mod`), so a fresh clone builds and tests with the
standard `go build ./...` / `go test ./...`, no local checkout or `replace`
needed. To co-develop against a local `simd` working copy, add a temporary
`replace github.com/tphakala/simd => ../simd` to `go.mod` (do not commit it).

## Packages

The library is split into subpackages so it can support multiple spectrogram
types over shared DSP primitives:

- `mel/` - SIMD log-mel spectrogram tensor generator (BSG-BAT preprocessing).
  Pipeline: periodic Hann window (`f32.Mul`) -> FFT -> power (`c64.AbsSq`) ->
  Slaney mel filterbank, 128 bins 9-150 kHz (`f32.DotProduct`) -> log10 +
  per-512-frame normalize (`f32.Mean/StdDev`). Produces the model input tensor
  `[512, 128]`. Import path is now `github.com/tphakala/go-spectrogram/mel`
  (moved from the module root).
- `sox/` - SoX-compatible spectrogram raster image renderer. Produces an
  `*image.Paletted` visually identical to `sox <in> -n spectrogram -r`
  (colormap, dB mapping, window/overlap math, dimensions). v1 is raster-only
  (no axes/legend), mono input, power-of-2 DFT, all SoX palette modes.
- `internal/dsp/` - shared radix-2 FFT (to be replaced by a simd kernel).
- `cmd/bench` - realtime-factor demo for the mel generator.

### sox usage

```go
import "github.com/tphakala/go-spectrogram/sox"

// samples: mono []float32 in [-1,1]; sampleRate in Hz.
err := sox.WritePNG("out.png", samples, 44100, sox.Options{})
// or get the image:
img, err := sox.Render(samples, 44100, sox.Options{Monochrome: true})
```

## Status / honesty (things to finish as this grows into a lib)

- Currently mel-specific and tied to the BSG-BAT parameters (384 kHz, n_fft=1024,
  hop=768, 128 mels, 9-150 kHz). Generalizing the params and adding a plain
  linear/STFT spectrogram is the obvious next step.
- Framing is `center=false` (streaming-friendly). librosa's default
  `center=true` adds n_fft/2 padding; matching it bit-for-bit is a follow-up, as
  is a golden-file parity test against a librosa reference.
- The mel filterbank is Slaney-normalized to match librosa defaults, but exact
  numerical parity vs librosa is not yet asserted.
- The `sox` package matches the installed SoX binary's raster visually (palette
  exact, per-pixel index within 1 of SoX); chrome (axes/labels/legend), multi-
  channel stacking, non-power-of-2 DFT, and Kaiser/Dolph windows are follow-ups.
