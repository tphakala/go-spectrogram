package sox

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

const pi = math.Pi

func mathSin(x float64) float64 { return math.Sin(x) }

// soxRunSpectrogram synthesizes nothing; it writes the given samples to a
// 32-bit float WAV and runs sox to produce a spectrogram PNG. When raw is
// true, -r is inserted (bare raster, no chrome). args are appended before -o.
func soxRunSpectrogram(t *testing.T, samples []float32, rate int, raw bool, args []string) string {
	t.Helper()
	dir := t.TempDir()
	wav := filepath.Join(dir, "in.wav")
	writeFloatWAV(t, wav, samples, rate)
	out := filepath.Join(dir, "out.png")
	full := []string{wav, "-n", "spectrogram"}
	if raw {
		full = append(full, "-r")
	}
	full = append(full, args...)
	full = append(full, "-o", out)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "sox", full...)
	cmd.Stderr = &bytes.Buffer{}
	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			t.Fatalf("sox timed out after 30s\n%s", cmd.Stderr.(*bytes.Buffer).String())
		}
		t.Fatalf("sox failed: %v\n%s", err, cmd.Stderr.(*bytes.Buffer).String())
	}
	return out
}

// soxPLTE runs sox on a short tone and returns the PLTE entries from the PNG.
func soxPLTE(t *testing.T, args []string) [][3]byte {
	t.Helper()
	samples := tone(8000, 0.5, 1000)
	png := soxRunSpectrogram(t, samples, 8000, true, args)
	data, err := os.ReadFile(png)
	if err != nil {
		t.Fatal(err)
	}
	return readPLTE(t, data)
}

// readPLTE extracts the PLTE chunk payload as RGB triples.
func readPLTE(t *testing.T, data []byte) [][3]byte {
	t.Helper()
	i := bytes.Index(data, []byte("PLTE"))
	if i < 4 {
		t.Fatal("no PLTE chunk")
	}
	length := int(binary.BigEndian.Uint32(data[i-4 : i]))
	if length <= 0 || length%3 != 0 || i+4+length > len(data) {
		t.Fatalf("invalid PLTE chunk: length=%d", length)
	}
	payload := data[i+4 : i+4+length]
	out := make([][3]byte, length/3)
	for k := range out {
		out[k] = [3]byte{payload[3*k], payload[3*k+1], payload[3*k+2]}
	}
	return out
}

// tone returns `dur` seconds of a sine at freq Hz, amplitude 0.5.
func tone(rate int, dur, freq float64) []float32 {
	n := int(float64(rate) * dur)
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(0.5 * mathSin(2*pi*freq*float64(i)/float64(rate)))
	}
	return s
}

// writeFloatWAV writes mono 32-bit IEEE float PCM so sox decodes the exact
// sample values we pass to Render.
func writeFloatWAV(t *testing.T, path string, samples []float32, rate int) {
	t.Helper()
	var buf bytes.Buffer
	dataLen := len(samples) * 4
	write := func(v any) { binary.Write(&buf, binary.LittleEndian, v) }
	buf.WriteString("RIFF")
	write(uint32(36 + dataLen))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	write(uint32(16))
	write(uint16(3)) // IEEE float
	write(uint16(1)) // mono
	write(uint32(rate))
	write(uint32(rate * 4)) // byte rate
	write(uint16(4))        // block align
	write(uint16(32))       // bits/sample
	buf.WriteString("data")
	write(uint32(dataLen))
	for _, s := range samples {
		write(s)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}
