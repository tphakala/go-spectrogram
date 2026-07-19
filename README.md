# go-spectrogram

A Go spectrogram library built on [github.com/tphakala/simd](https://github.com/tphakala/simd).

It started as a prototype mel-spectrogram generator for BSG-BAT (the
spectrogram-input bat CNN, [Zenodo 10.5281/zenodo.15495676](https://doi.org/10.5281/zenodo.15495676)),
written to answer "can the spectrogram run in realtime in Go" with a hard
in-language number and to scope what `simd` needs. It now supports multiple
spectrogram types: the original mel-tensor generator (`mel`) and a
SoX-compatible image renderer (`sox`). The two no longer share a transform:
`sox` runs the vendored real-input mixed-radix FFT in `internal/fft`, `mel` is
still on the full-size complex FFT in `internal/dsp`.

## Current results

### `sox`: faster than the SoX binary, and bit-exact against it

Rendering a 15 s mono clip at 24 kHz, at the four sizes a typical consumer
asks for, against `sox <in> -n spectrogram -x W -y H -d 15 -z 100`.

Compare the `WritePNG` column against SoX, not the `Render` column: `Render`
returns an in-memory `*image.Paletted`, whereas the binary also deflate-encodes
the PNG and writes it out, which is several ms on its own. The SoX timing
includes process spawn, since that is part of what invoking it costs.

`Render` uses every core by default (see `Options.Workers`); the `1 core`
column is the same call with `Workers: 1`, so the two together separate what
the per-core work costs from what parallelism buys.

amd64 (i7-1260P, AVX2+FMA), idle host, `performance` governor, 4 P-cores,
best of 6:

| size        | dft  | `Render` | `Render` 1 core | `WritePNG` | SoX 14.4.2 |
|-------------|------|----------|-----------------|------------|------------|
| 258 x 129   | 256  | 0.8 ms   | 2.1 ms          | **1.9 ms** | 5.4 ms     |
| 514 x 257   | 512  | 1.0 ms   | 2.4 ms          | **2.6 ms** | 7.0 ms     |
| 1026 x 513  | 1024 | 1.7 ms   | 4.5 ms          | **4.4 ms** | 12.9 ms    |
| 2050 x 1025 | 2048 | 5.6 ms   | 15.7 ms         | **12.3 ms**| 45.2 ms    |

arm64 (Raspberry Pi 5, Cortex-A76 at a pinned 2.4 GHz, Debian 13, Go 1.26.1),
the deployment target that matters most here, measured the same way:

| size        | `Render` | `Render` 1 core | `WritePNG` | SoX 14.4.2 |
|-------------|----------|-----------------|------------|------------|
| 258 x 129   | 2.1 ms   | 6.9 ms          | **4.1 ms** | 10.3 ms    |
| 514 x 257   | 2.6 ms   | 8.4 ms          | **6.1 ms** | 14.6 ms    |
| 1026 x 513  | 5.3 ms   | 16.9 ms         | **11.5 ms**| 30.6 ms    |
| 2050 x 1025 | 18.7 ms  | 63.7 ms         | **36.1 ms**| 112.6 ms   |

So `WritePNG` is 2.7-3.7x the SoX binary on amd64 and 2.4-3.1x on arm64. arm64
used to be the harder target, at a wash or worse; it no longer is, because the
same three things pay off on both: the vectorized butterfly has a NEON path,
the columns parallelise the same way, and the passes that were removed were
removed everywhere.

At the largest size `WritePNG` is now more than half PNG encoding: 12.3 ms
against 5.6 ms for `Render`. That is the stdlib encoder at its default
compression. A caller that would rather trade file size for latency can use
`Render` and encode itself with `png.BestSpeed`.

None of this cost any parity. Against the `sox` binary on the same host, every
case is 100.000% of pixels exact with worst delta 0, chrome and raster alike,
on both architectures, with the AVX2 kernels on amd64 and the NEON ones on
arm64. The one exception is unchanged and pre-dates all of this: at `DBRange`
180, 329 of 410400 pixels land one palette index off, identically on both
architectures. CI proves the rest rather than taking it on trust: the test job
runs on both `ubuntu-latest` and `ubuntu-24.04-arm`, and a further step re-runs
the comparison with `SIMD_DISABLE=all`, since simd picks a vectorized or scalar
kernel per host and the two do not agree to the last ulp.

That the vectorized butterfly changes nothing here is worth stating plainly: it
associates the arithmetic differently from the scalar radix-4 loop it replaced,
so it is not bit-identical to it, and the images still are.

Choosing float32 over float64 would not help here: measured on the Pi, simd's
f32 and f64 STFT plans came within 0.4% of each other at every transform size,
which is what you would expect while the butterfly stays scalar and lane count
never comes into play. float64 costs nothing measurable and is what matches
SoX's own double-precision `lsx_rdft`, so it is the right default. (This
package has no float32 variant to select; the comparison was between simd's
two plans.)

### Where the time goes now

Profiling one 2050 x 1025 render on a single core, after the work above:

| | share |
|---|---|
| `unravelPower` (real-FFT recombination, scalar) | 15% |
| butterflies (`f64.ButterflyComplex` + scalar radix-4) | 26% |
| frame load, window and digit reversal | 11% |
| palette quantisation | 10% |
| `f64.Log10` | 10% |
| raster transpose | 6% |

The transform is still over half the work. The two things that would move it
next both need simd kernels rather than anything in this repo, and are filed as
simd [#198](https://github.com/tphakala/simd/issues/198): a stage-level
butterfly (the current per-block API costs one call per block, so short spans
are dominated by call overhead) and a real-FFT unpack that writes `|X_k|^2`
directly. The latter is why `unravelPower` is still scalar: v1.6.0's
`RealFFTUnpack` is correct and vectorized and still measured 13% slower here,
because writing the complex spectrum and squaring it afterwards costs two more
passes over the bins than emitting power in one.

### `mel`: realtime bat preprocessing

- **8.9 ms** to compute the mel spectrogram for **1 second** of 384 kHz audio.
- **~106x realtime**, **0.94%** of one core, **zero allocations** in the hot path.
- Still on `internal/dsp`'s full-size complex FFT, so it pays roughly twice the
  transform that `sox` now does, and none of the `sox` work above. Moving it
  onto `internal/fft` is the obvious next speedup here.

```
go test ./...                              # correctness, plus SoX parity when sox is installed
go test -bench=. -benchmem -run=XXX        # benchmarks (vendored transform vs simd)
go run ./cmd/bench                         # friendly realtime number + active SIMD path
go test -run XXX -fuzz FuzzPowerInto ./internal/fft/   # transform invariants
```

The parity tests need the `sox` binary on PATH and skip without it. On a
hybrid-core host (Intel P/E), pin the benchmarks or the numbers are noise: an
unpinned run on the i7-1260P above swung by up to 69% between repeats. Pin the
CPU governor too; on a busy laptop the same benchmark varied by more than the
changes being measured.

```
taskset -c 0,2,4,6 env GOMAXPROCS=4 go test -bench=. -run=XXX -count=10 ./sox/
```

The module depends on [`github.com/tphakala/simd`](https://github.com/tphakala/simd)
(pinned to `v1.6.0` in `go.mod`), so a fresh clone builds and tests with the
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
- `internal/fft/` - vendored real-input transform used by `sox`. The transform
  dominates the render, and neither alternative was fast enough: `internal/dsp`
  runs a full-size complex FFT over real input, and simd's `f64.STFTPlan`
  halves that but leaves the butterfly scalar. It runs a scalar radix-4
  butterfly at the small spans and simd's vectorized `f64.ButterflyComplex`
  (added for [#192](https://github.com/tphakala/simd/issues/192)) from span 8
  up, which is where the kernel's per-call overhead stops mattering.
  Deliberately minimal (one frame, no framing or padding modes) so it can
  collapse back into a simd call once
  [#198](https://github.com/tphakala/simd/issues/198) lands.
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
- `mel` has had none of the `sox` optimisation work: no parallelism, no
  real-input transform, no fused quantisation.
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
