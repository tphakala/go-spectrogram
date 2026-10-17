# AGENTS.md

Orientation for coding agents and automated reviewers working on this repository. README.md is the user-facing reference (benchmarks, usage examples, status). This file is the map for changing and reviewing the code.

## What this is

`github.com/tphakala/go-spectrogram` is a pure-Go spectrogram library built on `github.com/tphakala/simd`. The root package (`doc.go`) exports nothing; callers use the subpackages. It aims to:

1. **Match SoX exactly where it claims to.** `sox` renders the same image as `sox <in> -n spectrogram`, chrome and raster alike, pixel for pixel. The installed `sox` binary is the oracle; the README states the one known exception (`DBRange` 180).
2. **Be faster than the SoX binary, especially on arm64.** The deployment targets are amd64 (AVX2/FMA) and arm64 (NEON, Raspberry Pi 5). Speed is a feature here, measured and recorded in the README.
3. **Run realtime for bat preprocessing.** `mel` produces the BSG-BAT model input tensor from 384 kHz audio with zero allocations in the hot path.

Packages:

| Package | Role |
|---|---|
| `sox` | SoX-compatible spectrogram image renderer: `Render`, `WritePNG`, `Options`, and `NewRenderer` (stateful `Renderer` that reuses buffers). Mono input, power-of-2 DFT, all SoX palette modes |
| `mel` | Log-mel spectrogram tensor generator (`NewGenerator`), hardcoded to the BSG-BAT parameters (384 kHz, n_fft 1024, hop 768, 128 mels) |
| `internal/fft` | Vendored real-input transform used by `sox` (scalar radix-4 at small spans, simd `f64.ButterflyComplex` from span 8) |
| `internal/dsp` | Radix-2 complex FFT still used by `mel`, plus shared helpers |
| `cmd/bench` | Realtime-factor demo for `mel` |

Non-goals (README "Status" lists them as follow-ups): multi-channel stacking, non-power-of-2 DFT, Kaiser and Dolph windows, resampling, librosa bit-parity for `mel`.

### API stability

The exported API of `sox` and `mel` is the contract. Do not change a signature, remove a symbol or change documented behaviour without calling it out as breaking. `Render` and `WritePNG` are safe for concurrent use; `Renderer` is not (see rule 5).

## Design rules you must not break

A change that violates one of these is wrong even if tests pass.

1. **The `sox` output stays bit-exact against the SoX binary.** The parity tests (`sox/parity_test.go`: `TestRasterParity`, `TestFullParity`) compare palette indices against `sox`. Any change to windowing, the transform, dB conversion, quantisation, palettes, geometry or chrome must leave them green. Do not loosen a tolerance or special-case a test to make it pass.
2. **Exactness must hold on both architectures and with and without SIMD.** simd picks a vectorized or scalar kernel per host and the two do not agree to the last ulp. CI runs the suite on amd64 and arm64 and re-runs the parity tests with `SIMD_DISABLE=all`. A change that is exact on one path only is a defect.
3. **float64 in the `sox` pipeline.** It matches SoX's own double-precision transform. Do not move it to float32 for speed; the README records the measurement that it buys nothing.
4. **The render path is allocation-sensitive and parallel.** Work is spread over `Options.Workers` goroutines (0 selects GOMAXPROCS). Results must not depend on the worker count (`sox/parallel_test.go`). Do not add per-column or per-pixel allocations, and keep buffers in the `Renderer` where they are reused.
5. **Reuse contract.** The image a `Renderer` returns aliases the buffer the next call reuses, and one `Renderer` belongs to one goroutine. `Render` allocates per call and stays concurrency-safe. `WritePNG` encodes before returning and never exposes the buffer. Keep README and godoc true when touching this.
6. **`WritePNG` is atomic.** It writes a unique temporary file in the destination directory and renames it into place, so the path either does not exist or holds a complete image. Preserve that.
7. **Validation order is a contract.** `Render` checks the sample rate before the `Options` (`TestRenderErrorPrecedence` pins it). Errors carry the `sox:` prefix; invalid `Options` return an error, never a panic.
8. **`mel` hot path allocates nothing.** Keep it that way; a generator is reused across calls.
9. **Pure Go on top of simd.** No cgo, and no dependency beyond `tphakala/simd`. The `sox` binary is a test oracle only, never a runtime dependency. `go.mod` pins the simd version; a temporary local `replace` for co-development must not be committed.
10. **`mel` and `sox` do not share a transform.** `sox` uses `internal/fft`, `mel` uses `internal/dsp`. Do not couple them without measuring both.

## Layout

```
doc.go               Umbrella package doc (no exported API)

sox/sox.go           Render, WritePNG (atomic temp file + rename)
sox/renderer.go      Renderer: NewRenderer, reusable buffers, parallel passes
sox/options.go       Options, validation, worker resolution
sox/analyzer.go      STFT analysis, dB conversion, per-worker scratch
sox/window.go        Window functions
sox/palette.go       SoX palettes and quantisation
sox/geometry.go      Image sizing and axis geometry
sox/chrome.go        Axes, tick labels, legend, title and comment
sox/font.go          SoX's embedded bitmap font
sox/*_test.go        Unit, parity (parity_test.go), oracle helpers (oracle_test.go), benchmarks

mel/mel.go           Generator, filterbank, normalisation
internal/fft/        Real-input transform, tests, FuzzPowerInto, benchmarks
internal/dsp/fft.go  Radix-2 FFT for mel
cmd/bench/main.go    Realtime demo
```

## Testing approach

- **Oracle tests need `sox` on PATH** and skip without it (`requireSox`). Some cases also skip on a SoX older than 14.4.3 for flags it lacks. CI installs `sox` from apt; locally, install it so the parity suite actually runs. A green run with the oracle tests skipped proves nothing about parity.
- **Regression tests.** New behaviour gets a test. A bug fix gets a regression test that fails before the fix. A test must be able to fail: break the line it pins and confirm it fails.
- **SIMD versus scalar.** `mel/simd_vs_scalar_test.go` compares the kernels against scalar code. Run parity with `SIMD_DISABLE=all` after touching a numeric path.
- **Fuzzing.** `FuzzPowerInto` in `internal/fft` checks transform invariants: `go test -run XXX -fuzz FuzzPowerInto ./internal/fft/`. CI does not run fuzz targets.
- **Benchmarks.** `sox/render_bench_test.go`, `internal/fft/bench_test.go`, and `go run ./cmd/bench`. On a hybrid-core host pin the CPUs or the numbers are noise, for example `taskset -c 0,2,4,6 env GOMAXPROCS=4 go test -bench=. -run=XXX -count=10 ./sox/`. Quote measured numbers, and re-measure before and after a change to a hot path.

## Commands

There is no Taskfile or lint config in this repo. The Go version is read from `go.mod` (currently 1.26).

```
go build ./...
go vet ./...
go test -race ./...                         # CI test step
SIMD_DISABLE=all go test -run Parity ./sox/ # CI scalar-fallback parity step
go test -bench=. -benchmem -run=XXX ./sox/
go run ./cmd/bench
gofmt -l .                                  # must print nothing
```

CI (`.github/workflows/ci.yml`) has a single `test` job on `ubuntu-latest` (amd64) and `ubuntu-24.04-arm` (arm64): install sox, `gofmt -l` check, `go vet`, `go build`, `go test -race ./...`, then the `SIMD_DISABLE=all` parity re-run. Actions are pinned by SHA.

## Conventions

- **Formatting.** `gofmt` clean; CI fails on any unformatted file.
- **Errors.** `sox` errors carry the `sox:` prefix and describe the offending option and value.
- **Comments.** Explain why: the SoX behaviour being matched, the failure avoided, the measurement behind a choice. Match the existing density.
- **Docs follow code.** A behaviour or performance change updates godoc and the README in the same change. The README carries measured numbers with the host and method; a number that no longer holds is a defect.
- **Commits** use an imperative subject with a type prefix (`feat:`, `fix:`, `docs:`, `ci:`, `perf:`, `test:`, `refactor:`), optionally scoped, with `!` for a breaking change.

## Review guidance

Rate findings on what a caller of the library would see.

Flag these, they are real defects here:

- Any break of the design rules above: a parity regression, a result that differs between SIMD and scalar paths, an unexpected allocation in a hot path, a new dependency or cgo.
- A parity test that was loosened, skipped or special-cased to pass, or a test that cannot fail on the behaviour it names.
- Output that depends on `Workers`, goroutine scheduling or the host CPU.
- A `Renderer` aliasing, ownership or concurrency guarantee that the code or docs no longer keep.
- A `WritePNG` path that can leave a partial file at the destination.
- A panic or index error reachable from `Options`, an empty or very short sample slice, or an odd sample rate, instead of a returned error.
- A README or godoc claim (performance, exactness, concurrency) that the change makes false.

Intentional, do not flag:

- The `DBRange` 180 exception: a few hundred pixels one palette index off, documented in the README.
- `mel` being hardcoded to the BSG-BAT parameters, using `center=false` framing, and having had none of the `sox` optimisation work.
- `sox` and `mel` using different transforms.
- The `sox` package doing no resampling, mono only, power-of-2 DFT only.
- Vendoring a minimal transform in `internal/fft` instead of calling simd's STFT plan.
- Parity tests skipping when `sox` is absent or too old.

## Things that do not belong in the repo

- Design docs, specs, plans, research notes and handoff prompts. `docs/superpowers/` is gitignored on purpose.
- Build outputs (`*.test`, `*.out`, `/bench`), profile files (`*.prof`, `*.pprof`), `go.work`, `.serena/` and `.codegraph/`.
- A committed `replace` directive pointing at a local simd checkout.
