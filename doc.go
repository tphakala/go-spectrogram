// Package gospectrogram is the umbrella for the go-spectrogram library.
//
// It does not export API itself; use the subpackages:
//
//   - mel  - SIMD log-mel spectrogram tensor generator (BSG-BAT preprocessing).
//   - sox  - SoX-compatible spectrogram PNG renderer (raster + axes/legend chrome).
//
// Shared low-level DSP primitives live in internal/dsp.
package gospectrogram
