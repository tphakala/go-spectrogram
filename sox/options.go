// Package sox renders a spectrogram image visually identical to SoX's
// `spectrogram` effect (colormap, dB mapping, window/overlap math, dimensions).
// By default it renders the full SoX PNG chrome (axes, tick labels, dBFS
// legend, footer comment, optional title); Options.Raw gives the bare raster.
// v1 is mono input, power-of-2 DFT.
package sox

import (
	"fmt"

	"github.com/tphakala/go-spectrogram/internal/dsp"
)

// WindowType selects the analysis window. v1 implements Hann, Hamming,
// Bartlett, Rectangular. Kaiser and Dolph are reserved for a follow-up and
// return an error from Render.
type WindowType int

const (
	WindowHann WindowType = iota // default
	WindowHamming
	WindowBartlett
	WindowRectangular
	WindowKaiser // v1: not implemented
	WindowDolph  // v1: not implemented
)

// Options mirrors SoX's spectrogram flags. The zero value yields SoX's default
// output; see normalize for the sentinel rules.
type Options struct {
	YSize      int // SoX -y: rows; dft_size = 2*(YSize-1). Must be power-of-2 dft in v1.
	YSizeTotal int // SoX -Y: total Y height; 0 => derive from 550.

	XSize        int     // SoX -x
	PixelsPerSec float64 // SoX -X
	Duration     float64 // SoX -d, seconds; 0 => len(samples)/sampleRate

	DBRange   int  // SoX -z; 0 => 120
	Gain      int  // SoX -Z; default 0
	Normalize bool // SoX -n

	Window       WindowType // default WindowHann
	WindowAdjust float64    // SoX -W (Kaiser/Dolph only; no effect in v1)
	SlackOverlap bool       // SoX -s

	Monochrome      bool // SoX -m
	HighColour      bool // SoX -h
	AltPalette      bool // SoX -A
	LightBackground bool // SoX -l
	Perm            int  // SoX -p, 1..6; 0 => 1
	Quantisation    int  // SoX -q, 1..249; 0 => 249

	Raw     bool   // SoX -r: render only the spectrogram raster, no axes/legend/text
	Title   string // SoX -t: title centred at the top (adds 20 rows); "" means no title (sox -t "" is inexpressible)
	Comment string // SoX -c: footer text at bottom-left; "" means "Created by SoX" (sox -c "" is inexpressible)
	NoAxes  bool   // SoX -a: no grid border lines, shorter ticks
}

// DefaultOptions returns the explicit SoX defaults (equivalent to the zero
// value after normalize).
func DefaultOptions() Options {
	return normalize(Options{})
}

// normalize fills "unset" fields (0/empty) with SoX defaults. The valid
// ranges of DBRange/Perm/Quantisation make 0 a safe sentinel; the empty
// string is safe for Comment since SoX always emits a default footer.
func normalize(o Options) Options {
	if o.DBRange == 0 {
		o.DBRange = 120
	}
	if o.Perm == 0 {
		o.Perm = 1
	}
	if o.Quantisation == 0 {
		o.Quantisation = 249
	}
	if o.Comment == "" {
		o.Comment = "Created by SoX"
	}
	return o
}

// validate checks a normalized Options. Returns an error mirroring SoX's
// getopts constraints plus the v1 restrictions.
func validate(o Options) error {
	if o.YSize != 0 && o.YSizeTotal != 0 {
		return fmt.Errorf("sox: only one of YSize / YSizeTotal may be set")
	}
	if b2i(o.XSize != 0)+b2i(o.PixelsPerSec != 0)+b2i(o.Duration != 0) > 2 {
		return fmt.Errorf("sox: at most two of XSize / PixelsPerSec / Duration may be set")
	}
	if o.XSize < 0 || o.PixelsPerSec < 0 || o.Duration < 0 || o.YSizeTotal < 0 {
		return fmt.Errorf("sox: XSize, PixelsPerSec, Duration, YSizeTotal must be non-negative")
	}
	if o.XSize != 0 && (o.XSize < 100 || o.XSize > maxXSize) {
		return fmt.Errorf("sox: XSize %d out of range 100..200000", o.XSize)
	}
	if o.PixelsPerSec != 0 && (o.PixelsPerSec < 1 || o.PixelsPerSec > 5000) {
		return fmt.Errorf("sox: PixelsPerSec %g out of range 1..5000", o.PixelsPerSec)
	}
	if o.YSizeTotal != 0 && (o.YSizeTotal < 130 || o.YSizeTotal > 200000) {
		return fmt.Errorf("sox: YSizeTotal %d out of range 130..200000", o.YSizeTotal)
	}
	if o.Window == WindowKaiser || o.Window == WindowDolph {
		return fmt.Errorf("sox: Kaiser/Dolph windows are not implemented in v1")
	}
	if o.Window < WindowHann || o.Window > WindowDolph {
		return fmt.Errorf("sox: invalid Window %d", o.Window)
	}
	if o.YSize != 0 {
		// SoX getopts bounds -y to [64, MAX_Y_SIZE] (200000 on 64-bit). Smaller
		// values would also break chrome drawing: the rotated frequency-axis
		// label needs more raster rows than a tiny YSize provides.
		if o.YSize < 64 || o.YSize > 200000 {
			return fmt.Errorf("sox: YSize %d out of range 64..200000", o.YSize)
		}
		if !dsp.IsPow2(2 * (o.YSize - 1)) {
			return fmt.Errorf("sox: YSize %d does not yield a power-of-2 dft_size (v1)", o.YSize)
		}
	}
	if o.DBRange < 20 || o.DBRange > 180 {
		return fmt.Errorf("sox: DBRange %d out of range 20..180", o.DBRange)
	}
	if o.Gain < -100 || o.Gain > 100 {
		return fmt.Errorf("sox: Gain %d out of range -100..100", o.Gain)
	}
	if o.Perm < 1 || o.Perm > 6 {
		return fmt.Errorf("sox: Perm %d out of range 1..6", o.Perm)
	}
	if o.Quantisation < 1 || o.Quantisation > 249 {
		return fmt.Errorf("sox: Quantisation %d out of range 1..249", o.Quantisation)
	}
	return nil
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
