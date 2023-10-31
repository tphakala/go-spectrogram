package main

import (
	"errors"
	"fmt"
	"image/color"
	"log"
	"math"
	"os"

	"github.com/fogleman/gg"
	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/mjibson/go-dsp/window"
	"gonum.org/v1/gonum/dsp/fourier"
)

const SampleRate = 48000

type ColorThreshold struct {
	Value float64
	Color color.RGBA
}

var baseColorPalette = []ColorThreshold{
	{-77.5, color.RGBA{0, 0, 45, 255}},
	{-75, color.RGBA{0, 0, 50, 255}},
	{-72.5, color.RGBA{0, 0, 55, 255}},
	{-70, color.RGBA{0, 0, 60, 255}},
	{-67.5, color.RGBA{0, 0, 65, 255}},
	{-65, color.RGBA{0, 0, 70, 255}},
	{-62.5, color.RGBA{0, 0, 75, 255}},
	{-60, color.RGBA{0, 0, 80, 255}},
	{-57.5, color.RGBA{0, 0, 85, 255}},
	{-55, color.RGBA{0, 0, 90, 255}},
	{-52.5, color.RGBA{0, 0, 95, 255}},
	{-50, color.RGBA{0, 0, 100, 255}},
	{-48.75, color.RGBA{0, 0, 105, 255}},
	{-47.5, color.RGBA{0, 0, 110, 255}},
	{-46.25, color.RGBA{0, 0, 115, 255}},
	{-45, color.RGBA{0, 0, 120, 255}},
	{-43.75, color.RGBA{0, 0, 124, 255}},
	{-42.5, color.RGBA{0, 0, 129, 255}},
	{-41.25, color.RGBA{10, 0, 134, 255}},
	{-40, color.RGBA{0, 0, 139, 255}},
	{-38.75, color.RGBA{19, 0, 139, 255}},
	{-37.5, color.RGBA{29, 0, 139, 255}},
	{-36.25, color.RGBA{34, 0, 137, 255}},
	{-35, color.RGBA{39, 0, 139, 255}},
	{-33.75, color.RGBA{45, 0, 137, 255}},
	{-32.5, color.RGBA{55, 0, 135, 255}},
	{-31.25, color.RGBA{62, 0, 133, 255}},
	{-30, color.RGBA{70, 0, 130, 255}},
	{-28.75, color.RGBA{77, 0, 129, 255}},
	{-27.5, color.RGBA{85, 0, 129, 255}},
	{-26.25, color.RGBA{92, 0, 128, 255}},
	{-25, color.RGBA{100, 0, 128, 255}},
	{-23.75, color.RGBA{109, 0, 128, 255}},
	{-22.5, color.RGBA{114, 0, 128, 255}},
	{-21.25, color.RGBA{121, 0, 125, 255}},
	{-20, color.RGBA{128, 0, 128, 255}},
	{-19.5, color.RGBA{133, 0, 124, 255}},
	{-19, color.RGBA{138, 0, 120, 255}},
	{-17.75, color.RGBA{143, 13, 97, 255}},
	{-16.5, color.RGBA{157, 26, 78, 255}},
	{-15.75, color.RGBA{161, 34, 60, 255}},
	{-15, color.RGBA{165, 42, 42, 255}},
	{-14.25, color.RGBA{176, 39, 39, 255}},
	{-13.5, color.RGBA{188, 37, 37, 255}},
	{-12.75, color.RGBA{199, 34, 34, 255}},
	{-12, color.RGBA{210, 32, 32, 255}},
	{-11.5, color.RGBA{221, 24, 24, 255}},
	{-11, color.RGBA{232, 16, 16, 255}},
	{-9.75, color.RGBA{243, 8, 8, 255}},
	{-10, color.RGBA{255, 0, 0, 255}},
	{-9.25, color.RGBA{255, 17, 0, 255}},
	{-8.5, color.RGBA{255, 34, 0, 255}},
	{-7.75, color.RGBA{255, 52, 0, 255}},
	{-7, color.RGBA{255, 69, 0, 255}},
	{-6.5, color.RGBA{255, 87, 0, 255}},
	{-6, color.RGBA{255, 105, 0, 255}},
	{-5.5, color.RGBA{255, 123, 0, 255}},
	{-5, color.RGBA{255, 140, 0, 255}},
	{-4.5, color.RGBA{255, 146, 0, 255}},
	{-4, color.RGBA{255, 152, 0, 255}},
	{-3.5, color.RGBA{255, 158, 0, 255}},
	{-3, color.RGBA{255, 165, 0, 255}},
	{-2.5, color.RGBA{255, 183, 0, 255}},
	{-2, color.RGBA{255, 210, 0, 255}},
	{-1.5, color.RGBA{255, 232, 0, 255}},
	{-1, color.RGBA{255, 255, 0, 255}},
	{-0.875, color.RGBA{255, 255, 32, 255}},
	{-0.75, color.RGBA{255, 255, 63, 255}},
	{-0.625, color.RGBA{255, 255, 95, 255}},
	{-0.5, color.RGBA{255, 255, 127, 255}},
	{-0.4, color.RGBA{255, 255, 151, 255}},
	{-0.3, color.RGBA{255, 255, 175, 255}},
	{-0.2, color.RGBA{255, 255, 200, 255}},
	{-0.1, color.RGBA{255, 255, 224, 255}},
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
func plotSpectrogram(pcm []float32, width, height, fftSize, hopSize int) *gg.Context {
	// Initialize FFT (Fast Fourier Transform) with the specified size.
	fft := fourier.NewFFT(fftSize)

	// Create a new graphics context with the specified width and height.
	dc := gg.NewContext(width, height)
	// Set the background color to black.
	dc.SetColor(color.RGBA{0, 0, 0, 255})
	dc.Clear()

	// Loop through the width of the spectrogram, which corresponds to time.
	for x := 0; x < width; x++ {
		// Determine start and end indices of the PCM data to be transformed.
		start := x * hopSize
		end := start + fftSize
		if end > len(pcm) {
			break
		}

		// Apply the Hann window function to the PCM data to smooth its edges.
		windowed := window.Hann(fftSize)
		src := make([]float64, fftSize)
		for i := start; i < end; i++ {
			src[i-start] = float64(pcm[i]) * windowed[i-start]
		}

		// Compute the FFT of the windowed data, yielding frequency coefficients.
		spectrum := fft.Coefficients(nil, src)

		// Loop through the spectrum magnitudes, convert them to dBFS (decibels relative to full scale),
		// and set the corresponding pixel color based on the magnitude.
		for y := 0; y < fftSize/2 && y < height; y++ {
			// Calculate the magnitude of the spectrum at the current frequency bin.
			mag := math.Sqrt(math.Pow(real(spectrum[y]), 2) + math.Pow(imag(spectrum[y]), 2))
			// Convert the magnitude to dBFS.
			dBFS := 20 * math.Log10((mag / 20))
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
func ReadAudioFile(filePath string) ([]float32, error) {
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
	var divisor float32
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
	var pcmData []float32
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
		// Convert each PCM sample to a float32 value and append it to the pcmData slice.
		for _, sample := range buf.Data[:n] {
			pcmData = append(pcmData, float32(sample)/divisor)
		}
	}

	// Notify that the reading process is done and indicate the number of samples read.
	fmt.Printf(", done, read %d samples\n", len(pcmData))
	return pcmData, nil
}

func main() {
	pcm, err := ReadAudioFile("tawnyowl.wav") // Your function that returns PCM data as []float32.
	if err != nil {
		log.Fatal(err)
	}
	width := len(pcm) / 1024 // Adjust as needed.
	height := 512            // Usually FFT size / 2.
	fftSize := 2048
	hopSize := 1024 // Adjust as needed, depending on overlap.

	dc := plotSpectrogram(pcm, width, height, fftSize, hopSize)
	dc.SavePNG("spectrogram.png")
}
