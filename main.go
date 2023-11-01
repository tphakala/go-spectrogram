package main

import (
	"errors"
	"fmt"
	"image/color"
	"log"
	"math"
	"math/cmplx"
	"os"

	"github.com/fogleman/gg"
	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/mjibson/go-dsp/fft"
	"github.com/mjibson/go-dsp/window"
)

const SampleRate = 48000

type ColorThreshold struct {
	Value float64
	Color color.RGBA
}

var baseColorPalette = []ColorThreshold{
	{-120, color.RGBA{0, 0, 0, 255}},       // black
	{-117.5, color.RGBA{0, 0, 17, 255}},    // very very very dark blue
	{-115, color.RGBA{0, 0, 34, 255}},      // very very dark blue
	{-112.5, color.RGBA{0, 0, 51, 255}},    // deeper dark blue
	{-110, color.RGBA{0, 0, 69, 255}},      // very dark blue
	{-107.5, color.RGBA{0, 0, 86, 255}},    // deeper blue
	{-105, color.RGBA{0, 0, 104, 255}},     // darker blue
	{-102.5, color.RGBA{0, 0, 121, 255}},   // more dark blue
	{-100, color.RGBA{0, 0, 139, 255}},     // dark blue
	{-97.5, color.RGBA{0, 0, 155, 255}},    // intermediate dark blue
	{-95, color.RGBA{0, 0, 172, 255}},      // medium dark blue
	{-92.5, color.RGBA{0, 0, 188, 255}},    // brighter dark blue
	{-90, color.RGBA{0, 0, 205, 255}},      // medium blue
	{-87.5, color.RGBA{0, 0, 218, 255}},    // medium bright blue
	{-85, color.RGBA{0, 0, 230, 255}},      // brighter blue
	{-82.5, color.RGBA{0, 0, 242, 255}},    // much brighter blue
	{-80, color.RGBA{0, 0, 255, 255}},      // blue
	{-77.5, color.RGBA{19, 0, 223, 255}},   // deep blue-indigo
	{-75, color.RGBA{38, 0, 192, 255}},     // indigo-ish
	{-72.5, color.RGBA{57, 0, 161, 255}},   // deeper indigo
	{-70, color.RGBA{75, 0, 130, 255}},     // indigo
	{-67.5, color.RGBA{94, 0, 150, 255}},   // indigo-violet mix
	{-65, color.RGBA{112, 0, 171, 255}},    // dark violet-ish
	{-62.5, color.RGBA{130, 0, 191, 255}},  // darker violet
	{-60, color.RGBA{148, 0, 211, 255}},    // dark violet
	{-57.5, color.RGBA{146, 0, 193, 255}},  // violet-ish
	{-55, color.RGBA{144, 0, 175, 255}},    // medium violet
	{-52.5, color.RGBA{142, 0, 157, 255}},  // less violet
	{-50, color.RGBA{139, 0, 139, 255}},    // dark magenta
	{-47.5, color.RGBA{168, 0, 104, 255}},  // magenta-red mix
	{-45, color.RGBA{197, 0, 69, 255}},     // magenta-red-ish
	{-42.5, color.RGBA{226, 0, 34, 255}},   // deep red
	{-40, color.RGBA{255, 0, 0, 255}},      // red
	{-37.5, color.RGBA{255, 18, 0, 255}},   // deep red-orange
	{-35, color.RGBA{255, 35, 0, 255}},     // red-orange mix
	{-32.5, color.RGBA{255, 52, 0, 255}},   // more orange than red
	{-30, color.RGBA{255, 69, 0, 255}},     // red-orange
	{-27.5, color.RGBA{255, 93, 0, 255}},   // orange-ish
	{-25, color.RGBA{255, 117, 0, 255}},    // deeper orange
	{-22.5, color.RGBA{255, 141, 0, 255}},  // less deep orange
	{-20, color.RGBA{255, 165, 0, 255}},    // orange
	{-17.5, color.RGBA{255, 188, 0, 255}},  // light orange
	{-15, color.RGBA{255, 210, 0, 255}},    // brighter light orange
	{-12.5, color.RGBA{255, 233, 0, 255}},  // very light orange
	{-10, color.RGBA{255, 255, 0, 255}},    // yellow
	{-7.5, color.RGBA{255, 255, 64, 255}},  // light yellow
	{-5, color.RGBA{255, 255, 128, 255}},   // very light yellow
	{-2.5, color.RGBA{255, 255, 192, 255}}, // pale yellow
	{0, color.RGBA{255, 255, 255, 255}},    // white
}

// interpolateColor interpolates between two colors (c1 and c2) based on a given fraction.
// It linearly interpolates each RGB channel of the two colors. The alpha channel is set to 255.
// For example, a fraction of 0.5 will give a color halfway between c1 and c2.
func interpolateColor(c1, c2 color.RGBA, fraction float64) color.RGBA {
	return color.RGBA{
		// Interpolate the red channel.
		uint8(float64(c1.R) + fraction*(float64(c2.R)-float64(c1.R))),
		// Interpolate the green channel.
		uint8(float64(c1.G) + fraction*(float64(c2.G)-float64(c1.G))),
		// Interpolate the blue channel.
		uint8(float64(c1.B) + fraction*(float64(c2.B)-float64(c1.B))),
		// Set alpha channel to maximum (opaque).
		255,
	}
}

// generateFineGrainedPalette takes a base palette of ColorThresholds and interpolates
// to create a more fine-grained palette. This provides smoother color transitions.
func generateFineGrainedPalette(base []ColorThreshold) []ColorThreshold {
	var fineGrainedPalette []ColorThreshold

	// Iterate through the base palette. For each pair of consecutive colors,
	// add the first color, then an interpolated color halfway between the pair.
	for i := 0; i < len(base)-1; i++ {
		// Append the current color from the base palette.
		fineGrainedPalette = append(fineGrainedPalette, base[i])

		// Calculate the average value between the current and next threshold.
		interpolatedValue := (base[i].Value + base[i+1].Value) / 2
		// Interpolate a color halfway between the current and next color.
		interpolatedColor := interpolateColor(base[i].Color, base[i+1].Color, 0.5)
		// Append the interpolated color and value.
		fineGrainedPalette = append(fineGrainedPalette, ColorThreshold{interpolatedValue, interpolatedColor})
	}
	// Append the last color from the base palette to the fine-grained palette.
	fineGrainedPalette = append(fineGrainedPalette, base[len(base)-1])

	// Return the newly generated fine-grained palette.
	return fineGrainedPalette
}

// Generate a fine-grained color palette based on the baseColorPalette.
var colorPalette = generateFineGrainedPalette(baseColorPalette)

// getColorForDBFS returns the appropriate color for a given dBFS value by
// checking against predefined color thresholds in the colorPalette.
func getColorForDBFS(dBFS float64) color.RGBA {
	// Iterate through each color threshold in the palette.
	for _, threshold := range colorPalette {
		// If the given dBFS value is less than or equal to the threshold's value,
		// return the threshold's color.
		if dBFS <= threshold.Value {
			return threshold.Color
		}
	}
	// Default to white color if the dBFS value doesn't match any threshold.
	return color.RGBA{255, 255, 255, 255}
}

// plotSpectrogram takes PCM audio data and visualizes it as a spectrogram.
// The resulting spectrogram represents the frequency content of the PCM data over time.
func plotSpectrogram(pcm []float64, width, height, fftSize, hopSize int) *gg.Context {
	// Create a new graphics context with the specified width and height.
	dc := gg.NewContext(width, height)
	// Set the background color to black.
	dc.SetColor(color.RGBA{0, 0, 0, 255})
	dc.Clear()

	// Calculate the total energy in the Hann window function.
	windowFunc := window.Hann(fftSize)
	windowEnergy := 0.0
	for _, w := range windowFunc {
		windowEnergy += w * w
	}

	// Loop through the width of the spectrogram, which corresponds to time.
	for x := 0; x < width; x++ {
		// Determine start and end indices of the PCM data to be transformed.
		start := x * hopSize
		end := start + fftSize
		if end > len(pcm) {
			break
		}

		// Apply the Hann window function to the PCM data to smooth its edges.
		src := make([]float64, fftSize)
		for i := start; i < end; i++ {
			src[i-start] = pcm[i] * windowFunc[i-start] // Windowed data
		}

		// Compute the FFT of the windowed data, yielding frequency coefficients.
		spectrum := fft.FFTReal(src)

		// Loop through the spectrum magnitudes, convert them to dBFS, and set the pixel color based on the magnitude.
		for y := 0; y < fftSize/2 && y < height; y++ {
			// Calculate the magnitude of the spectrum at the current frequency bin.
			mag := cmplx.Abs(complex128(spectrum[y]))
			// Convert the magnitude to dBFS.
			dBFS := (20 * math.Log10(mag/math.Sqrt(windowEnergy))) - 10
			fmt.Println(windowEnergy)
			//			if dBFS > -8 {
			//fmt.Println(dBFS)
			//}
			// Set the pixel color based on its dBFS value.
			dc.SetColor(getColorForDBFS(dBFS))
			// Draw the pixel on the graphics context.
			dc.SetPixel(x, height-y-1)
		}
	}

	// Return the graphics context containing the drawn spectrogram.
	return dc
}

// ReadAudioFile reads an audio file from the specified path and returns its data as a slice of float32 values.
func ReadAudioFile(filePath string) ([]float64, error) {
	// Notify that the reading process has begun.
	fmt.Print("- Reading audio data")

	// Open the audio file.
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	// Ensure the file is closed after all operations are done.
	defer file.Close()

	// Create a new WAV file decoder.
	decoder := wav.NewDecoder(file)
	// Read the audio file's meta information.
	decoder.ReadInfo()
	// Check if the provided audio file is a valid WAV format.
	if !decoder.IsValidFile() {
		return nil, errors.New("input is not a valid WAV audio file")
	}

	// This block is for debug purposes; prints details of the WAV file.
	if false {
		fmt.Println("File is valid wav: ", decoder.IsValidFile())
		fmt.Println("Sample rate:", decoder.SampleRate)
		fmt.Println("Bits per sample:", decoder.BitDepth)
		fmt.Println("Channels:", decoder.NumChans)
	}

	// Ensure the audio file has the expected sample rate.
	if decoder.SampleRate != SampleRate {
		return nil, errors.New("input file sample rate is not valid")
	}

	// Determine the divisor for converting audio samples based on the bit depth.
	var divisor float64
	switch decoder.BitDepth {
	case 16:
		divisor = 32768.0
	case 24:
		divisor = 8388608.0
	case 32:
		divisor = 2147483648.0
	default:
		return nil, errors.New("unsupported audio file bit depth")
	}

	// Slice for holding the PCM audio data.
	var pcmData []float64
	// Initialize a buffer to read the PCM data.
	buf := &audio.IntBuffer{Data: make([]int, SampleRate), Format: &audio.Format{SampleRate: int(SampleRate), NumChannels: 1}}

	// Read and convert the PCM audio data from the file.
	for {
		// Read a chunk of PCM data into the buffer.
		n, err := decoder.PCMBuffer(buf)
		if err != nil {
			return nil, err
		}
		// If no data is read, end the loop.
		if n == 0 {
			break
		}
		// Convert each PCM sample to a float64 value and append it to the pcmData slice.
		for _, sample := range buf.Data[:n] {
			pcmData = append(pcmData, float64(sample)/divisor)
		}
	}

	// Notify that the reading process is done and indicate the number of samples read.
	fmt.Printf(", done, read %d samples\n", len(pcmData))
	return pcmData, nil
}

func computeDCOffset(samples []float64) float64 {
	var sum float64
	for _, sample := range samples {
		sum += float64(sample)
	}
	return sum / float64(len(samples))
}

func computeMinMaxLevel(samples []float64) (float64, float64) {
	minLevel := samples[0]
	maxLevel := samples[0]

	for _, sample := range samples {
		if sample < minLevel {
			minLevel = sample
		}
		if sample > maxLevel {
			maxLevel = sample
		}
	}

	return minLevel, maxLevel
}

func computePkLevDB(maxLevel float64) float64 {
	return 20 * math.Log10(math.Abs(maxLevel))
}

func main() {
	pcm, err := ReadAudioFile("tawnyowl.wav") // Your function that returns PCM data as []float32.
	if err != nil {
		log.Fatal(err)
	}

	dcOffset := computeDCOffset(pcm)
	minLevel, maxLevel := computeMinMaxLevel(pcm)
	pkLevDB := computePkLevDB(maxLevel)

	fmt.Printf("DC offset   %.6f\n", dcOffset)
	fmt.Printf("Min level   %.2f\n", minLevel)
	fmt.Printf("Max level   %.2f\n", maxLevel)
	fmt.Printf("Pk lev dB   %.2f\n", pkLevDB)

	width := len(pcm) / 880 // Adjust as needed.
	height := 512           // Usually FFT size / 2.
	fftSize := 2048
	hopSize := 880 // Adjust as needed, depending on overlap.

	dc := plotSpectrogram(pcm, width, height, fftSize, hopSize)
	dc.SavePNG("spectrogram.png")
}
