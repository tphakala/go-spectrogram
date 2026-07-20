package sox

import (
	"image"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// rendererCases is the Options matrix every reuse test runs over. Each entry
// exercises a different piece of the state a Renderer carries between calls:
// raw vs chrome decides whether the canvas has bytes no pass rewrites,
// normalize decides which analysis buffer is used and is the only path where a
// stale per-image maximum can bite, a title changes the canvas geometry, and
// Workers: 1 takes the serial branch, which is a different reset path from the
// fan-out one.
var rendererCases = []struct {
	name string
	opt  Options
}{
	{"chrome", Options{XSize: 258, YSize: 129}},
	{"raw", Options{XSize: 258, YSize: 129, Raw: true}},
	{"normalize", Options{XSize: 258, YSize: 129, Normalize: true}},
	{"normalize_raw", Options{XSize: 258, YSize: 129, Raw: true, Normalize: true}},
	{"title", Options{XSize: 258, YSize: 129, Title: "reuse"}},
	{"serial", Options{XSize: 258, YSize: 129, Workers: 1}},
	{"larger", Options{XSize: 514, YSize: 257}},
	{"light_mono", Options{XSize: 258, YSize: 129, Monochrome: true, LightBackground: true}},
	{"multiblock", Options{XSize: 258, YSize: 129, PixelsPerSec: 10}},
}

type reuseClip struct {
	name    string
	samples []float32
	rate    float64
}

// reuseClips is ordered to make stale state visible rather than to be
// representative. Loud is followed by quiet because a maximum left over from a
// previous render only changes the image when the next one is quieter, and the
// lengths jump around because a stale frame position or an unsliced schedule
// only shows up when the column count changes between calls. The last clip is
// short enough to yield fewer columns than there are workers.
func reuseClips() []reuseClip {
	tone := func(n int, rate, amp float64) []float32 {
		s := make([]float32, n)
		for i := range s {
			t := float64(i) / rate
			s[i] = float32(amp * (math.Sin(2*math.Pi*1000*t) + 0.3*math.Sin(2*math.Pi*4300*t)))
		}
		return s
	}
	return []reuseClip{
		{"loud_5s", tone(5*24000, 24000, 0.9), 24000},
		{"quiet_5s", tone(5*24000, 24000, 0.002), 24000},
		{"long_12s", tone(12*24000, 24000, 0.5), 24000},
		{"short_1s", tone(1*24000, 24000, 0.5), 24000},
		{"other_rate_3s", tone(3*16000, 16000, 0.5), 16000},
		{"tiny", tone(700, 24000, 0.5), 24000},
	}
}

// assertSameImage compares two paletted images byte for byte. The palette is
// compared too: it is cached on the Renderer, so a bug that shared one palette
// across differing Options would only be visible here.
func assertSameImage(t *testing.T, ctx string, got, want *image.Paletted) {
	t.Helper()
	if got.Rect != want.Rect {
		t.Fatalf("%s: rect = %v, want %v", ctx, got.Rect, want.Rect)
	}
	if got.Stride != want.Stride {
		t.Fatalf("%s: stride = %d, want %d", ctx, got.Stride, want.Stride)
	}
	if len(got.Palette) != len(want.Palette) {
		t.Fatalf("%s: palette len = %d, want %d", ctx, len(got.Palette), len(want.Palette))
	}
	for i := range want.Palette {
		if got.Palette[i] != want.Palette[i] {
			t.Fatalf("%s: palette[%d] = %v, want %v", ctx, i, got.Palette[i], want.Palette[i])
		}
	}
	if len(got.Pix) != len(want.Pix) {
		t.Fatalf("%s: pix len = %d, want %d", ctx, len(got.Pix), len(want.Pix))
	}
	for i := range want.Pix {
		if got.Pix[i] != want.Pix[i] {
			n := 0
			for j := range want.Pix {
				if got.Pix[j] != want.Pix[j] {
					n++
				}
			}
			t.Fatalf("%s: pix differs at %d (%d != %d), %d of %d bytes total",
				ctx, i, got.Pix[i], want.Pix[i], n, len(want.Pix))
		}
	}
}

// TestRendererMatchesRender is the test that matters: a Renderer reused over a
// sequence of different clips must produce exactly what a fresh Render produces
// for each of them. Every stale-state bug the reuse introduces (a frame buffer
// still primed from the previous clip, a maximum carried over under -n, a
// schedule appended to rather than reset, chrome ghosting through an uncleared
// canvas) shows up as a wrong image and nothing else, so it is worth doing this
// over the whole Options matrix rather than one representative case.
func TestRendererMatchesRender(t *testing.T) {
	clips := reuseClips()
	for _, tc := range rendererCases {
		t.Run(tc.name, func(t *testing.T) {
			r, err := NewRenderer(tc.opt)
			if err != nil {
				t.Fatalf("NewRenderer: %v", err)
			}
			for _, c := range clips {
				want, err := Render(c.samples, c.rate, tc.opt)
				if err != nil {
					t.Fatalf("%s: Render: %v", c.name, err)
				}
				got, err := r.Render(c.samples, c.rate)
				if err != nil {
					t.Fatalf("%s: Renderer.Render: %v", c.name, err)
				}
				assertSameImage(t, c.name, got, want)
			}
		})
	}
}

// TestRendererRepeatedIdenticalClip pins the simplest form of the same
// property. Rendering one clip twice through the same Renderer must give the
// same image both times; a buffer that is accumulated into rather than
// overwritten would pass the sequence test above by luck but fail here.
func TestRendererRepeatedIdenticalClip(t *testing.T) {
	clip := reuseClips()[0]
	for _, tc := range rendererCases {
		t.Run(tc.name, func(t *testing.T) {
			r, err := NewRenderer(tc.opt)
			if err != nil {
				t.Fatalf("NewRenderer: %v", err)
			}
			first, err := r.Render(clip.samples, clip.rate)
			if err != nil {
				t.Fatalf("first: %v", err)
			}
			// Copied because the next call overwrites the pixels in place, which
			// is exactly the documented aliasing this test must not trip over.
			snapshot := &image.Paletted{
				Pix:     append([]uint8(nil), first.Pix...),
				Stride:  first.Stride,
				Rect:    first.Rect,
				Palette: first.Palette,
			}
			second, err := r.Render(clip.samples, clip.rate)
			if err != nil {
				t.Fatalf("second: %v", err)
			}
			assertSameImage(t, "second render", second, snapshot)
		})
	}
}

// TestRendererReturnsFreshHeader documents the aliasing contract in the
// direction that is safe to rely on: the returned *image.Paletted is a new
// header every call, so holding on to an earlier one cannot make its Rect or
// Stride change underneath the caller. Only the Pix bytes are reused.
func TestRendererReturnsFreshHeader(t *testing.T) {
	clips := reuseClips()
	// Driven by PixelsPerSec, not XSize: with an explicit XSize the column count
	// is that size whatever the clip's duration, so the image geometry would
	// never change and the test would not be testing anything.
	r, err := NewRenderer(Options{PixelsPerSec: 50, YSize: 129})
	if err != nil {
		t.Fatalf("NewRenderer: %v", err)
	}
	a, err := r.Render(clips[0].samples, clips[0].rate)
	if err != nil {
		t.Fatalf("render a: %v", err)
	}
	rectA := a.Rect
	b, err := r.Render(clips[3].samples, clips[3].rate)
	if err != nil {
		t.Fatalf("render b: %v", err)
	}
	if a == b {
		t.Fatal("Render returned the same *image.Paletted header twice")
	}
	if a.Rect != rectA {
		t.Fatalf("earlier header mutated: rect = %v, want %v", a.Rect, rectA)
	}
	if b.Rect == rectA {
		t.Fatal("test is not exercising a geometry change; both clips gave the same rect")
	}
}

// bytesPerRender returns the bytes f allocates per call, averaged over n calls
// after one warm-up. TotalAlloc is cumulative and unaffected by collection, so
// this measures allocation rather than retained heap, which is the thing a
// reuse API is meant to change.
func bytesPerRender(n int, f func()) uint64 {
	var before, after runtime.MemStats
	f()
	runtime.GC()
	runtime.ReadMemStats(&before)
	for range n {
		f()
	}
	runtime.ReadMemStats(&after)
	return (after.TotalAlloc - before.TotalAlloc) / uint64(n)
}

// TestRendererSteadyStateAllocs is the point of the whole change.
//
// It asserts on bytes rather than on allocation count, because bytes are what
// the Renderer removes: the buffers it holds on to are three large ones, while
// most of the remaining count is chrome formatting its tick labels and the
// per-worker closures, neither of which this change touches. Both bounds are
// loose on purpose, so that a later refactor moving a couple of allocations
// does not fail a test whose subject is megabytes.
func TestRendererSteadyStateAllocs(t *testing.T) {
	clip := reuseClips()[0]
	opt := Options{XSize: 514, YSize: 257}

	r, err := NewRenderer(opt)
	if err != nil {
		t.Fatalf("NewRenderer: %v", err)
	}
	render := func() {
		if _, err := r.Render(clip.samples, clip.rate); err != nil {
			t.Fatalf("reused: %v", err)
		}
	}
	free := func() {
		if _, err := Render(clip.samples, clip.rate, opt); err != nil {
			t.Fatalf("fresh: %v", err)
		}
	}

	reusedBytes, freshBytes := bytesPerRender(20, render), bytesPerRender(20, free)
	reusedAllocs := testing.AllocsPerRun(5, render)
	freshAllocs := testing.AllocsPerRun(5, free)
	t.Logf("per render: reused %d B / %.0f allocs, fresh %d B / %.0f allocs",
		reusedBytes, reusedAllocs, freshBytes, freshAllocs)

	if reusedBytes*10 > freshBytes {
		t.Errorf("reused Renderer allocates %d B/render against %d B for a fresh Render; "+
			"expected at least a 10x reduction", reusedBytes, freshBytes)
	}
	if reusedAllocs*2 > freshAllocs {
		t.Errorf("reused Renderer makes %.0f allocations/render against %.0f for a fresh Render; "+
			"expected at least a 2x reduction", reusedAllocs, freshAllocs)
	}
}

// TestRendererValidatesOptionsOnce checks that construction is where bad
// Options are rejected, so a caller that builds a Renderer at startup learns
// about a misconfiguration then rather than on the first clip.
func TestRendererValidatesOptionsOnce(t *testing.T) {
	if _, err := NewRenderer(Options{DBRange: 200}); err == nil {
		t.Fatal("NewRenderer accepted DBRange 200")
	}
	if _, err := NewRenderer(Options{XSize: 10}); err == nil {
		t.Fatal("NewRenderer accepted XSize 10")
	}
	r, err := NewRenderer(Options{XSize: 258, YSize: 129})
	if err != nil {
		t.Fatalf("NewRenderer: %v", err)
	}
	for _, rate := range []float64{0, -1, math.NaN()} {
		if _, err := r.Render([]float32{0, 1, 0}, rate); err == nil {
			t.Errorf("Renderer.Render accepted sampleRate %g", rate)
		}
	}
}

// TestRendererWritePNG covers the path a batch consumer actually uses, where
// the image never escapes and the aliasing is therefore invisible.
func TestRendererWritePNG(t *testing.T) {
	opt := Options{XSize: 258, YSize: 129}
	r, err := NewRenderer(opt)
	if err != nil {
		t.Fatalf("NewRenderer: %v", err)
	}
	dir := t.TempDir()
	for i, c := range reuseClips()[:3] {
		viaRenderer := filepath.Join(dir, "r"+string(rune('a'+i))+".png")
		viaFunc := filepath.Join(dir, "f"+string(rune('a'+i))+".png")
		if err := r.WritePNG(viaRenderer, c.samples, c.rate); err != nil {
			t.Fatalf("%s: Renderer.WritePNG: %v", c.name, err)
		}
		if err := WritePNG(viaFunc, c.samples, c.rate, opt); err != nil {
			t.Fatalf("%s: WritePNG: %v", c.name, err)
		}
		got, err := os.ReadFile(viaRenderer)
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		want, err := os.ReadFile(viaFunc)
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if string(got) != string(want) {
			t.Errorf("%s: Renderer.WritePNG produced a different file from WritePNG", c.name)
		}
	}
}
