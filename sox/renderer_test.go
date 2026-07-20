package sox

import (
	"image"
	// Registered explicitly rather than relying on sox.go importing it, so that
	// image.Decode in assertDecodesTo does not depend on a non-test file's
	// import set.
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
)

// rendererCases is the Options matrix every reuse test runs over. Each entry
// exercises a different piece of the state a Renderer carries between calls:
// raw vs chrome decides whether the canvas has bytes no pass rewrites,
// normalize decides which analysis buffer is used and is the only path where a
// stale per-image maximum can bite, a title changes the canvas geometry, and
// Workers: 1 takes the serial branch, which is a different reset path from the
// fan-out one.
//
// The last three entries exist for properties the rest do not reach, each
// verified rather than assumed:
//
//   - normalize_multiblock is the only case whose dBFS buffer has to GROW on a
//     reused Renderer. With an explicit XSize the column count is that size for
//     every clip, so it can only shrink; driving the time axis with
//     PixelsPerSec instead makes it track duration (measured over this corpus:
//     50, 50, 120, 10, 30, 20, 1, 0 columns, so the 12 s clip grows it twice).
//     Without this case, a grow helper that never grew a live buffer passed the
//     whole package.
//   - few_cols pins the partial fan-out, where some workers get an empty range.
//     Reachability of that branch otherwise depends on the machine: it needs
//     more than about six workers, and both CI runners have four.
//   - multiblock keeps a case where a column spans several DFTs on a clip long
//     enough to have many columns.
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
	{"normalize_multiblock", Options{YSize: 129, PixelsPerSec: 10, Normalize: true}},
	{"few_cols", Options{YSize: 129, PixelsPerSec: 10, Workers: 8}},
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
// only shows up when the column count changes between calls.
//
// The tail is the awkward inputs: a clip whose samples are not finite, which is
// what could drive the -n maximum somewhere pathological; one shorter than a
// single DFT frame; and an empty one, which renders zero columns and is the
// only input that reaches the early return in analyze. Note that "short" does
// not by itself mean "few columns": with an explicit XSize, resolveTimeAxis
// saturates pixels-per-second and the 29 ms clip still yields 175 columns. The
// few-columns property belongs to the few_cols and multiblock Options entries,
// not to any clip.
func reuseClips() []reuseClip {
	tone := func(n int, rate, amp float64) []float32 {
		s := make([]float32, n)
		for i := range s {
			t := float64(i) / rate
			s[i] = float32(amp * (math.Sin(2*math.Pi*1000*t) + 0.3*math.Sin(2*math.Pi*4300*t)))
		}
		return s
	}
	nonFinite := tone(2*24000, 24000, 0.5)
	nonFinite[100] = float32(math.NaN())
	nonFinite[200] = float32(math.Inf(1))
	nonFinite[300] = float32(math.Inf(-1))

	return []reuseClip{
		{"loud_5s", tone(5*24000, 24000, 0.9), 24000},
		{"quiet_5s", tone(5*24000, 24000, 0.002), 24000},
		{"long_12s", tone(12*24000, 24000, 0.5), 24000},
		{"short_1s", tone(1*24000, 24000, 0.5), 24000},
		{"other_rate_3s", tone(3*16000, 16000, 0.5), 16000},
		{"non_finite_2s", nonFinite, 24000},
		{"tiny", tone(700, 24000, 0.5), 24000},
		{"empty", nil, 24000},
	}
}

// assertSameImage compares two paletted images byte for byte, including the
// palette: a Renderer caches one and hands the same slice to every image it
// returns, so a palette built from the wrong Options would only be visible here.
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
//
// Note what it can and cannot catch. Since Render is itself implemented as
// NewRenderer(...).Render(...), both sides run the same single-render code, so
// this is a differential between a virgin instance and a reused one: it pins
// everything asymmetric about reuse and nothing about the render itself. The
// independent oracle for the render is parity_test.go, which compares against
// the SoX 14.4.2 binary bit for bit on two architectures.
func TestRendererMatchesRender(t *testing.T) {
	t.Parallel()
	clips := reuseClips()
	for _, tc := range rendererCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
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
	t.Parallel()
	clip := reuseClips()[0]
	for _, tc := range rendererCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
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
// Stride change underneath the caller. Only the pixels are reused.
//
// It also pins that the pixel slice is capped to its length, so a caller's
// append allocates a copy instead of writing into the Renderer's live buffer.
func TestRendererReturnsFreshHeader(t *testing.T) {
	t.Parallel()
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
	// Checked on the SECOND, smaller render, which is the only place it can
	// fail: the first render allocates exactly n, so its capacity equals its
	// length whether or not the returned slice is capped. Only after a shrink
	// does the buffer carry spare capacity for an append to escape into.
	if cap(b.Pix) != len(b.Pix) {
		t.Errorf("Pix cap %d exceeds len %d; a caller's append would write into "+
			"the Renderer's live pixels", cap(b.Pix), len(b.Pix))
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

// perRender returns the bytes and allocations f averages per call, over n calls
// after one warm-up. TotalAlloc and Mallocs are cumulative counters unaffected
// by collection, so this measures allocation rather than retained heap, which is
// the thing a reuse API is meant to change.
//
// Both figures deliberately come from the same mechanism. testing.AllocsPerRun
// would be the obvious way to count, but it pins GOMAXPROCS to 1 for its
// duration, so a fresh Render measured through it fans out over one worker while
// the Renderer being compared against was built for the ambient core count. That
// is not a like-for-like comparison, and it reads as one.
func perRender(n int, f func()) (bytes, allocs uint64) {
	var before, after runtime.MemStats
	f()
	runtime.GC()
	runtime.ReadMemStats(&before)
	for range n {
		f()
	}
	runtime.ReadMemStats(&after)
	return (after.TotalAlloc - before.TotalAlloc) / uint64(n),
		(after.Mallocs - before.Mallocs) / uint64(n)
}

// TestRendererSteadyStateAllocs is the point of the whole change.
//
// The bytes bound is the real assertion: the buffers are what the Renderer
// holds, and they are three orders of magnitude larger than everything else.
// The allocation count gets a much looser bound because most of what remains is
// chrome formatting its tick labels and the per-worker closures, neither of
// which this change touches. Both are ratios between two measurements taken the
// same way in the same process, so neither depends on machine speed or load.
//
// Not run in parallel with anything: MemStats is process-wide.
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

	reusedBytes, reusedAllocs := perRender(20, render)
	freshBytes, freshAllocs := perRender(20, free)
	t.Logf("per render: reused %d B / %d allocs, fresh %d B / %d allocs",
		reusedBytes, reusedAllocs, freshBytes, freshAllocs)

	// 20x rather than 10x: a regression that stopped reusing only the schedule
	// slices still came in at 12x, so the looser bound let it through.
	if reusedBytes*20 > freshBytes {
		t.Errorf("reused Renderer allocates %d B/render against %d B for a fresh Render; "+
			"expected at least a 20x reduction", reusedBytes, freshBytes)
	}
	if reusedAllocs*2 > freshAllocs {
		t.Errorf("reused Renderer makes %d allocations/render against %d for a fresh Render; "+
			"expected at least a 2x reduction", reusedAllocs, freshAllocs)
	}
}

// TestRendererValidatesOptionsAtConstruction checks that bad Options are
// rejected by the constructor, so a caller building a Renderer at startup learns
// about a misconfiguration then rather than on the first clip, and that the
// error is the same one Render would have given for the same Options.
func TestRendererValidatesOptionsAtConstruction(t *testing.T) {
	t.Parallel()
	sig := reuseClips()[0].samples
	for _, bad := range []Options{{DBRange: 200}, {XSize: 10}, {Workers: -1}} {
		_, viaNew := NewRenderer(bad)
		if viaNew == nil {
			t.Errorf("NewRenderer accepted %+v", bad)
			continue
		}
		_, viaRender := Render(sig, 24000, bad)
		if viaRender == nil {
			t.Errorf("Render accepted %+v", bad)
			continue
		}
		if viaNew.Error() != viaRender.Error() {
			t.Errorf("%+v: NewRenderer says %q, Render says %q", bad, viaNew, viaRender)
		}
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

// TestRenderErrorPrecedence pins the ordering the free Render documents: a bad
// sample rate is reported as such even when the Options are also invalid, so the
// error a caller sees does not depend on which of their two mistakes is checked
// first. NewRenderer deliberately orders it the other way, since it has no
// sample rate to check.
func TestRenderErrorPrecedence(t *testing.T) {
	t.Parallel()
	badOpts := Options{DBRange: 200}
	err := func() error {
		_, err := Render([]float32{0, 1, 0}, 0, badOpts)
		return err
	}()
	if err == nil {
		t.Fatal("Render accepted a zero sample rate with invalid Options")
	}
	if !strings.Contains(err.Error(), "sampleRate") {
		t.Errorf("Render reported %q, want the sampleRate error to take precedence", err)
	}
	if err := WritePNG(filepath.Join(t.TempDir(), "x.png"), []float32{0, 1, 0}, 0, badOpts); err == nil {
		t.Error("WritePNG accepted a zero sample rate")
	} else if !strings.Contains(err.Error(), "sampleRate") {
		t.Errorf("WritePNG reported %q, want the sampleRate error", err)
	}
	if _, err := NewRenderer(badOpts); err == nil {
		t.Error("NewRenderer accepted DBRange 200")
	} else if !strings.Contains(err.Error(), "DBRange") {
		t.Errorf("NewRenderer reported %q, want the DBRange error", err)
	}
}

// TestRenderConcurrent pins the claim that the free Render stays safe for
// concurrent use now that it is built on a stateful type.
//
// Each goroutine renders a DIFFERENT clip, which is what makes the image
// comparison able to fail: if the entry points ever came to share a buffer,
// eight goroutines all rendering the same clip would write identical bytes over
// each other and compare equal, leaving -race as the only detector. With
// distinct clips a shared buffer produces a visibly wrong image as well.
func TestRenderConcurrent(t *testing.T) {
	t.Parallel()
	opt := Options{XSize: 258, YSize: 129}
	clips := reuseClips()[:5] // the five that render a non-empty image
	want := make([]*image.Paletted, len(clips))
	for i, c := range clips {
		img, err := Render(c.samples, c.rate, opt)
		if err != nil {
			t.Fatalf("reference %s: %v", c.name, err)
		}
		want[i] = img
	}

	const perClip = 3
	n := len(clips) * perClip
	got := make([]*image.Paletted, n)
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c := clips[i%len(clips)]
			got[i], errs[i] = Render(c.samples, c.rate, opt)
		}()
	}
	wg.Wait()
	for i := range n {
		if errs[i] != nil {
			t.Fatalf("goroutine %d: %v", i, errs[i])
		}
		assertSameImage(t, "concurrent "+clips[i%len(clips)].name, got[i], want[i%len(clips)])
	}
}

// TestRendererWritePNG covers the path a batch consumer actually uses, where
// the image never escapes and the aliasing is therefore invisible. It runs the
// whole clip corpus so that the geometry changes between writes, and decodes one
// result so that a shared encoder regression cannot pass by producing two
// identically wrong files.
func TestRendererWritePNG(t *testing.T) {
	t.Parallel()
	opt := Options{XSize: 258, YSize: 129}
	r, err := NewRenderer(opt)
	if err != nil {
		t.Fatalf("NewRenderer: %v", err)
	}
	dir := t.TempDir()
	for i, c := range reuseClips() {
		viaRenderer := filepath.Join(dir, "r"+string(rune('a'+i))+".png")
		viaFunc := filepath.Join(dir, "f"+string(rune('a'+i))+".png")

		errRenderer := r.WritePNG(viaRenderer, c.samples, c.rate)
		errFunc := WritePNG(viaFunc, c.samples, c.rate, opt)
		// The empty clip renders zero columns, which the PNG encoder rejects.
		// Both entry points must agree on that, which is worth asserting rather
		// than skipping.
		if (errRenderer == nil) != (errFunc == nil) {
			t.Fatalf("%s: Renderer.WritePNG err = %v, WritePNG err = %v", c.name, errRenderer, errFunc)
		}
		if errFunc != nil {
			continue
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
			continue
		}
		assertDecodesTo(t, c, viaRenderer, r)
	}
}

// assertDecodesTo checks the written file really is the image the Renderer
// produced, rather than two entry points agreeing on the same wrong bytes.
func assertDecodesTo(t *testing.T, c reuseClip, path string, r *Renderer) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("%s: open: %v", c.name, err)
	}
	defer func() {
		if err := f.Close(); err != nil {
			t.Errorf("%s: close: %v", c.name, err)
		}
	}()
	decoded, _, err := image.Decode(f)
	if err != nil {
		t.Fatalf("%s: decode: %v", c.name, err)
	}
	pal, ok := decoded.(*image.Paletted)
	if !ok {
		t.Fatalf("%s: decoded to %T, want *image.Paletted", c.name, decoded)
	}
	fresh, err := r.Render(c.samples, c.rate)
	if err != nil {
		t.Fatalf("%s: re-render: %v", c.name, err)
	}
	if pal.Rect != fresh.Rect {
		t.Fatalf("%s: decoded rect %v, want %v", c.name, pal.Rect, fresh.Rect)
	}
	for y := range fresh.Rect.Dy() {
		for x := range fresh.Rect.Dx() {
			if pal.ColorIndexAt(x, y) != fresh.ColorIndexAt(x, y) {
				t.Fatalf("%s: decoded index at (%d,%d) = %d, want %d",
					c.name, x, y, pal.ColorIndexAt(x, y), fresh.ColorIndexAt(x, y))
			}
		}
	}
}
