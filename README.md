# go-spectrogram

A Go spectrogram library built on [github.com/tphakala/simd](https://github.com/tphakala/simd).

It started as a prototype mel-spectrogram generator for BSG-BAT (the
spectrogram-input bat CNN, [Zenodo 10.5281/zenodo.15495676](https://doi.org/10.5281/zenodo.15495676)),
written to answer "can the spectrogram run in realtime in Go" with a hard
in-language number and to scope what `simd` needs. It is now the foundation for a
fuller spectrogram library.

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

Run with `GOFLAGS=-mod=mod GOPROXY=off`. The module uses a local `replace` for
`simd` (see `go.mod`) so it co-develops against the working `simd` checkout at
`../simd`.

## Layout

- `mel.go` - mel-spectrogram generator. Pipeline: periodic Hann window
  (`f32.Mul`) -> FFT -> power (`c64.AbsSq`) -> Slaney mel filterbank, 128 bins
  9-150 kHz (`f32.DotProduct`) -> log10 + per-512-frame normalize
  (`f32.Mean/StdDev`). Produces the model input tensor `[512, 128]`.
- `fft.go` - hand-rolled radix-2 FFT, to be replaced by a simd kernel.
- `cmd/bench` - realtime-factor demo.

## Status / honesty (things to finish as this grows into a lib)

- Currently mel-specific and tied to the BSG-BAT parameters (384 kHz, n_fft=1024,
  hop=768, 128 mels, 9-150 kHz). Generalizing the params and adding a plain
  linear/STFT spectrogram is the obvious next step.
- Framing is `center=false` (streaming-friendly). librosa's default
  `center=true` adds n_fft/2 padding; matching it bit-for-bit is a follow-up, as
  is a golden-file parity test against a librosa reference.
- The mel filterbank is Slaney-normalized to match librosa defaults, but exact
  numerical parity vs librosa is not yet asserted.
- Root package is `mel`; a fuller lib will likely move it to a `mel/` subpackage
  with the module root as the umbrella.
