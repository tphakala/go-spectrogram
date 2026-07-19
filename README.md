# go-spectrogram

A Go spectrogram library built on [github.com/tphakala/simd](https://github.com/tphakala/simd).

It started as a prototype mel-spectrogram generator for BSG-BAT (the
spectrogram-input bat CNN, [Zenodo 10.5281/zenodo.15495676](https://doi.org/10.5281/zenodo.15495676)),
written to answer "can the spectrogram run in realtime in Go" with a hard
in-language number and to scope what `simd` needs. It now supports multiple
spectrogram types: the original mel-tensor generator (`mel`) and a
SoX-compatible image renderer (`sox`), over a shared DSP core.

## Current results (i7-1260P, AVX2+FMA)

### `sox`: faster than the SoX binary, and bit-exact against it

Rendering a 15 s mono clip at 24 kHz, at the four sizes a typical consumer
asks for, against `sox <in> -n spectrogram -x W -y H -d 15 -z 100`.

Compare the `WritePNG` column against SoX, not the `Render` column: `Render`
returns an in-memory `*image.Paletted`, whereas the binary also deflate-encodes
the PNG and writes it out, which is several ms on its own. The SoX timing
includes process spawn, since that is part of what invoking it costs.

| size        | dft  | `Render` | `WritePNG` | SoX 14.4.2 |
|-------------|------|----------|------------|------------|
| 258 x 129   | 256  | 5.5 ms   | **6.5 ms** | 6 ms       |
| 514 x 257   | 512  | 6.2 ms   | **8.0 ms** | 8 ms       |
| 1026 x 513  | 1024 | 10.0 ms  | **13.8 ms**| 16 ms      |
| 2050 x 1025 | 2048 | 40.4 ms  | **45.5 ms**| 61 ms      |

So end to end it is roughly a wash at the two small sizes and about 1.2-1.3x
faster at the two large ones. The bigger practical win is not the milliseconds:
it is losing the subprocess, and with it the spawn cost, the OOM-killer
exposure on small machines, the pipe plumbing, and the `sox` path configuration.

`WritePNG` uses the stdlib encoder at its default compression, which is about
3.4 ms of the 13.8 ms at 1026 x 513. A caller that would rather trade file size
for latency can use `Render` and encode itself: `png.BestSpeed` cuts that to
1.5 ms at the cost of roughly 9 KiB -> 14 KiB per image.

Every rendered pixel matches the binary exactly: the parity tests report
**100.000% exact, worst delta 0** across palette modes, dynamic ranges, gains,
overlap settings, and the drain/truncation edge cases, chrome included.

Most of the remaining time is the DFT: `f64.STFTPlan`'s butterfly is still
scalar Go, which is 50% of a default render. simd
[#192](https://github.com/tphakala/simd/issues/192) tracks vectorizing it.

### `mel`: realtime bat preprocessing

- **8.9 ms** to compute the mel spectrogram for **1 second** of 384 kHz audio.
- **~106x realtime**, **0.94%** of one core, **zero allocations** in the hot path.
- Not yet moved onto the fused `f32.STFTPlan` + `DotProductBatch` path the way
  `sox` was; that is the obvious next speedup here.

```
go test ./...                              # correctness (FFT tone, shapes, normalize range)
go test -bench=. -benchmem -run=XXX        # benchmarks (incl. simd vs scalar)
go run ./cmd/bench                         # friendly realtime number + active SIMD path
```

The module depends on [`github.com/tphakala/simd`](https://github.com/tphakala/simd)
(pinned to `v1.5.0` in `go.mod`), so a fresh clone builds and tests with the
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
- `sox/` - SoX-compatible spectrogram image renderer. Produces an
  `*image.Paletted` matching `sox <in> -n spectrogram`: full chrome (axes,
  tick labels, dBFS legend, title/comment, SoX's embedded bitmap font) by
  default, bare raster via `Raw: true` (`sox ... -r`). Mono input,
  power-of-2 DFT, all SoX palette modes.
- `internal/dsp/` - radix-2 FFT still used by `mel`, plus shared helpers. `sox`
  now uses simd's `f64.STFTPlan` instead.
- `cmd/bench` - realtime-factor demo for the mel generator.

### sox usage

```go
import "github.com/tphakala/go-spectrogram/sox"

// samples: mono []float32 in [-1,1]; sampleRate in Hz.
err := sox.WritePNG("out.png", samples, 44100, sox.Options{})        // full SoX PNG
img, err := sox.Render(samples, 44100, sox.Options{Raw: true})       // raster only
img, err = sox.Render(samples, 44100, sox.Options{Title: "My clip"}) // with title
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
- The `sox` package matches the installed SoX binary exactly, chrome and raster
  alike (100.000% of palette indices, worst delta 0). Multi-channel stacking,
  non-power-of-2 DFT, and Kaiser/Dolph windows are follow-ups. Note that
  `-w dolph` is a hard gap for any consumer that uses SoX's `scientific` styles,
  and the package does no resampling, so a caller that relies on `rate` in the
  SoX effect chain has to resample before calling `Render`.
