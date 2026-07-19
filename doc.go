// Package gospectrogram is the umbrella for the go-spectrogram library.
//
// It does not export API itself; use the subpackages:
//
//   - mel  - SIMD log-mel spectrogram tensor generator (BSG-BAT preprocessing).
//   - sox  - SoX-compatible spectrogram PNG renderer (raster + axes/legend chrome).
//
// The sox renderer's transform lives in internal/fft; internal/dsp holds the
// FFT that mel still uses, plus shared helpers.
package gospectrogram
