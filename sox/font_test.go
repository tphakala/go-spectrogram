package sox

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

// Inflated font ground truth, computed from the fixed[] blob in
// sox v14.4.2 spectrogram.c.
const wantFontSHA256 = "936673ea8c3b1fa4efc5b05198ba0d232da0017f6cde1f9d6336b131533a0172"

func TestFontDataInflates(t *testing.T) {
	f := fontData()
	if len(f) != 96*fontY {
		t.Fatalf("font length %d, want %d", len(f), 96*fontY)
	}
	sum := sha256.Sum256(f)
	if got := hex.EncodeToString(sum[:]); got != wantFontSHA256 {
		t.Errorf("font sha256 %s, want %s", got, wantFontSHA256)
	}
}

func TestFontGlyphZero(t *testing.T) {
	// Glyph for '0' (index '0'-' ' = 16), 12 row bytes, top row first.
	want := [12]byte{0x00, 0x20, 0x50, 0x88, 0x88, 0x88, 0x88, 0x88, 0x50, 0x20, 0x00, 0x00}
	f := fontData()
	got := f[16*fontY : 17*fontY]
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("glyph '0' row %d = %#02x, want %#02x", i, got[i], want[i])
		}
	}
}
