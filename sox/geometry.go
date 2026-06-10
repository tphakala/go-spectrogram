package sox

import "math"

const maxXSize = 200000

// deriveDFTSize returns (dft_size, rows). rows = dft_size/2 + 1.
// Ports spectrogram.c:408-430 for the mono, power-of-2 case.
func deriveDFTSize(o Options) (dft, rows int) {
	if o.YSize != 0 {
		dft = 2 * (o.YSize - 1)
	} else {
		yt := o.YSizeTotal
		if yt == 0 {
			yt = 550
		}
		y := yt - 2 // mono: channels = 1
		if y < 32 {
			y = 32
		}
		for dft = 128; dft <= y; dft <<= 1 {
		}
	}
	return dft, dft/2 + 1
}

// resolveTimeAxis ports the x_size / pixels_per_sec resolution loop
// (spectrogram.c:383-406). duration is the audio length in seconds (always
// known for in-memory input). Returns (xSize, pixelsPerSec).
func resolveTimeAxis(o Options, duration float64) (xSize int, pixelsPerSec float64) {
	xSize = o.XSize
	pixelsPerSec = o.PixelsPerSec
	d := duration
	if o.Duration != 0 {
		d = o.Duration
	}
	for {
		switch {
		case pixelsPerSec == 0 && xSize != 0 && d != 0:
			pixelsPerSec = math.Min(5000, float64(xSize)/d)
		case xSize == 0 && pixelsPerSec != 0 && d != 0:
			xSize = int(math.Min(maxXSize, pixelsPerSec*d+0.5))
		}
		switch {
		case xSize == 0:
			xSize = 800
			continue
		case pixelsPerSec == 0:
			pixelsPerSec = 100
			continue
		}
		break
	}
	return xSize, pixelsPerSec
}

// stepSizing ports spectrogram.c:431-439. actual is the raw window sum from
// makeWindow(ws, 0). Returns (stepSize, blockSteps, blockNorm). All divisions
// are done in float64 to avoid Go integer-division truncating away Ceil/Floor.
func stepSizing(actual float64, dftSize int, rate, pixelsPerSec float64, slack bool) (stepSize, blockSteps int, blockNorm float64) {
	base := actual
	if slack {
		base = math.Sqrt(actual * float64(dftSize))
	}
	stepSize = int(base + 0.5)
	// Defensive clamps (no-ops for valid SoX inputs, where the window sum keeps
	// stepSize in [1, dftSize]): a zero stepSize would make the division below
	// +Inf and int(+Inf) is implementation-defined in Go.
	if stepSize < 1 {
		stepSize = 1
	}
	blockSteps = int(math.Max(rate/pixelsPerSec, 1))
	stepSize = int(float64(blockSteps)/math.Ceil(float64(blockSteps)/float64(stepSize)) + 0.5)
	if stepSize < 1 {
		stepSize = 1
	} else if stepSize > dftSize {
		stepSize = dftSize
	}
	blockSteps = int(math.Floor(float64(blockSteps)/float64(stepSize) + 0.5))
	if blockSteps < 1 {
		blockSteps = 1
	}
	blockNorm = 1.0 / float64(blockSteps)
	return stepSize, blockSteps, blockNorm
}
