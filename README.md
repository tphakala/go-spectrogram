# go-spectrogram

A Go spectrogram library built on [github.com/tphakala/simd](https://github.com/tphakala/simd).

It started as a prototype mel-spectrogram generator for BSG-BAT (the
spectrogram-input bat CNN, [Zenodo 10.5281/zenodo.15495676](https://doi.org/10.5281/zenodo.15495676)),
written to answer "can the spectrogram run in realtime in Go" with a hard
in-language number and to scope what `simd` needs. It now supports multiple
spectrogram types: the original mel-tensor generator (`mel`) and a
SoX-compatible image renderer (`sox`). The two no longer share a transform:
`sox` runs the vendored real-input radix-4 FFT in `internal/fft`, `mel` is
still on the full-size complex FFT in `internal/dsp`.

## Current results (i7-1260P, AVX2+FMA)

### `sox`: faster than the SoX binary, and bit-exact against it

Rendering a 15 s mono clip at 24 kHz, at the four sizes a typical consumer
asks for, against `sox <in> -n spectrogram -x W -y H -d 15 -z 100`.

Compare the `WritePNG` column against SoX, not the `Render` column: `Render`
returns an in-memory `*image.Paletted`, whereas the binary also deflate-encodes
the PNG and writes it out, which is several ms on its own. The SoX timing
includes process spawn, since that is part of what invoking it costs.

amd64 (i7-1260P), best of 12, pinned to P-cores:

| size        | dft  | `Render` | `WritePNG` | SoX 14.4.2 |
|-------------|------|----------|------------|------------|
| 258 x 129   | 256  | 2.7 ms   | **4.0 ms** | 5 ms       |
| 514 x 257   | 512  | 3.9 ms   | **5.7 ms** | 6 ms       |
| 1026 x 513  | 1024 | 6.8 ms   | **9.3 ms** | 13 ms      |
| 2050 x 1025 | 2048 | 28.0 ms  | **34.8 ms**| 51 ms      |

`WritePNG` uses the stdlib encoder at its default compression, which is about
2.5 ms of the 9.3 ms at 1026 x 513. A caller that would rather trade file size
for latency can use `Render` and encode itself: `png.BestSpeed` cuts that to
1.5 ms at the cost of roughly 9 KiB -> 14 KiB per image.

### arm64

The same comparison on a Raspberry Pi 5 (Cortex-A76, Debian 13, Go 1.26.1),
the deployment target that matters most here:

| size        | `WritePNG` | SoX 14.4.2 |
|-------------|------------|------------|
| 258 x 129   | 9.5 ms     | **9 ms**   |
| 514 x 257   | 14.3 ms    | **14 ms**  |
| 1026 x 513  | **28.3 ms**| 30 ms      |
| 2050 x 1025 | **109.6 ms**| 112 ms    |

Parity holds identically here, including the `DBRange` 180 exception above,
which produces the same 329 mismatches on both architectures. CI now proves that
rather than taking it on trust: the test job runs on both `ubuntu-latest` and
`ubuntu-24.04-arm`, so the NEON `Log10` kernel is held to exactly the same
assertions as the AVX2 one. A further step re-runs the parity comparison with
`SIMD_DISABLE=all`, since simd picks a vectorized or scalar kernel per host and
the two do not agree to the last ulp.

arm64 is the harder target. Before the transform was vendored, SoX won by
1.2-1.5x at every size here; it is now a wash at the small sizes and a small
win at the large ones. `internal/fft` is the reason, and further gains need
simd [#192](https://github.com/tphakala/simd/issues/192) to vectorize the
butterfly.

Choosing float32 over float64 would not help here: measured on the Pi, simd's
f32 and f64 STFT plans came within 0.4% of each other at every transform size,
which is what you would expect while the butterfly stays scalar and lane count
never comes into play. float64 costs nothing measurable and is what matches
SoX's own double-precision `lsx_rdft`, so it is the right default. (This
package has no float32 variant to select; the comparison was between simd's
two plans.)

### `mel`: realtime bat preprocessing

- **8.9 ms** to compute the mel spectrogram for **1 second** of 384 kHz audio.
- **~106x realtime**, **0.94%** of one core, **zero allocations** in the hot path.
- Still on `internal/dsp`'s full-size complex FFT, so it pays roughly twice the
  transform that `sox` now does. Moving it onto a real-input transform is the
  obvious next speedup here.

```
go test ./...                              # correctness, plus SoX parity when sox is installed
go test -bench=. -benchmem -run=XXX        # benchmarks (vendored transform vs simd)
go run ./cmd/bench                         # friendly realtime number + active SIMD path
go test -run XXX -fuzz FuzzPowerInto ./internal/fft/   # transform invariants
```

The parity tests need the `sox` binary on PATH and skip without it. On a
hybrid-core host (Intel P/E), pin the benchmarks or the numbers are noise: an
unpinned run on the i7-1260P above swung by up to 69% between repeats.

```
taskset -c 0,2,4,6 env GOMAXPROCS=4 go test -bench=. -run=XXX -count=10 ./sox/
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
- `internal/fft/` - vendored radix-4 real-input transform used by `sox`. The
  transform dominates the render, and neither alternative was fast enough:
  `internal/dsp` runs a full-size complex FFT over real input, and simd's
  `f64.STFTPlan` halves that but leaves the butterfly scalar. simd
  [#192](https://github.com/tphakala/simd/issues/192) tracks the f64 kernels
  that would make this package unnecessary. Deliberately minimal (one frame, no
  framing or padding modes) so it can collapse back into a simd call.
- `internal/dsp/` - radix-2 FFT still used by `mel`, plus shared helpers.
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

- `mel` is hardcoded to the BSG-BAT parameters (384 kHz, n_fft=1024, hop=768,
  128 mels, 9-150 kHz). Generalizing those and adding a plain linear/STFT
  spectrogram is the obvious next step. (`sox` is fully parameterised via
  `Options`; this bullet is about `mel` only.)
- Framing is `center=false` (streaming-friendly). librosa's default
  `center=true` adds n_fft/2 padding; matching it bit-for-bit is a follow-up, as
  is a golden-file parity test against a librosa reference.
- The mel filterbank is Slaney-normalized to match librosa defaults, but exact
  numerical parity vs librosa is not yet asserted.
- The `sox` package matches the installed SoX binary bit-exactly, chrome and
  raster alike, except at `DBRange` 180 where 329 of 410400 pixels land one
  palette index off (pre-existing, identical on the float32 FFT this replaced;
  see the parity results above). Multi-channel stacking,
  non-power-of-2 DFT, and Kaiser/Dolph windows are follow-ups. Note that
  `-w dolph` is a hard gap for any consumer that uses SoX's `scientific` styles,
  and the package does no resampling, so a caller that relies on `rate` in the
  SoX effect chain has to resample before calling `Render`.
