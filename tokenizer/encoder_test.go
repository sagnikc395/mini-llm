package main

import (
	"bytes"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"testing/quick"
)

func trainOn(t *testing.T, text string, numMerges int) *Encoder {
	t.Helper()
	toks, freqs := CountTypes([]byte(text), 4)
	tr := NewTrainer(toks, freqs, 4, 64)
	return NewEncoder(tr.Train(numMerges, nil))
}

// The byte-level base vocab is what guarantees this for *arbitrary* bytes,
// including invalid UTF-8 — hence quick over []byte rather than string.
func TestRoundTripArbitraryBytes(t *testing.T) {
	enc := trainOn(t, randomText(rand.New(rand.NewSource(3)), 200_000), 400)
	f := func(b []byte) bool {
		return bytes.Equal(enc.Decode(enc.Encode(b, 1)), b)
	}
	if err := quick.Check(f, &quick.Config{MaxCount: 1000}); err != nil {
		t.Fatal(err)
	}
	for _, w := range []int{2, 8} {
		g := func(b []byte) bool { return bytes.Equal(enc.Decode(enc.Encode(b, w)), b) }
		if err := quick.Check(g, &quick.Config{MaxCount: 300}); err != nil {
			t.Fatalf("workers=%d: %v", w, err)
		}
	}
}

func TestRoundTripText(t *testing.T) {
	text := randomText(rand.New(rand.NewSource(5)), 300_000)
	enc := trainOn(t, text, 600)
	got := enc.Decode(enc.Encode([]byte(text), 8))
	if !bytes.Equal(got, []byte(text)) {
		t.Fatalf("round trip lost data: %d bytes in, %d out", len(text), len(got))
	}
}

// Chunking must not change the token stream: boundaries land after newlines
// and merges never cross pre-token boundaries.
func TestEncodeParallelMatchesSerial(t *testing.T) {
	text := randomText(rand.New(rand.NewSource(9)), 400_000)
	enc := trainOn(t, text, 500)
	want := enc.Encode([]byte(text), 1)
	for _, w := range []int{2, 3, 8, 16, 64} {
		if got := enc.Encode([]byte(text), w); !reflect.DeepEqual(got, want) {
			t.Fatalf("workers=%d: token stream differs from serial (%d vs %d tokens)", w, len(got), len(want))
		}
	}
}

// Encoding applies merges in rank order, not greedily left to right.
func TestEncodeUsesRankOrder(t *testing.T) {
	// Force the merges: "ab" first, then "bc". Encoding "abc" must produce
	// [ab][c] because (a,b) has the lower rank, even though a greedy scan
	// that preferred the longer match might pick differently.
	enc := NewEncoder([]Merge{
		{'a', 'b', 256},
		{'b', 'c', 257},
	})
	got := enc.Encode([]byte("abc"), 1)
	if !reflect.DeepEqual(got, []uint32{256, 'c'}) {
		t.Fatalf("got %v, want [256 99]", got)
	}
	if !bytes.Equal(enc.Decode(got), []byte("abc")) {
		t.Fatal("round trip failed")
	}
}

func TestCacheAgreesWithUncached(t *testing.T) {
	text := randomText(rand.New(rand.NewSource(13)), 100_000)
	enc := trainOn(t, text, 300)
	buf := make([]uint32, 0, 64)
	s := scanner{b: []byte(text)}
	var want []uint32
	for {
		lo, hi, ok := s.next()
		if !ok {
			break
		}
		buf = enc.encodeToken([]byte(text)[lo:hi], buf)
		want = append(want, buf...)
	}
	if got := enc.Encode([]byte(text), 4); !reflect.DeepEqual(got, want) {
		t.Fatal("cached encoding differs from a fresh encode per pre-token")
	}
}

func TestModelAndTokenRoundTrip(t *testing.T) {
	text := randomText(rand.New(rand.NewSource(17)), 100_000)
	enc := trainOn(t, text, 200)
	ids := enc.Encode([]byte(text), 4)

	dir := t.TempDir()
	mp := filepath.Join(dir, "model.json")
	if err := SaveModel(mp, enc.merges); err != nil {
		t.Fatal(err)
	}
	merges, err := LoadModel(mp)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(merges, enc.merges) {
		t.Fatal("model did not survive the round trip")
	}

	for _, bin := range []bool{false, true} {
		tp := filepath.Join(dir, "tokens")
		if err := SaveTokens(tp, ids, bin); err != nil {
			t.Fatal(err)
		}
		back, err := LoadTokens(tp, bin)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(back, ids) {
			t.Fatalf("binary=%v: tokens did not survive the round trip", bin)
		}
		if !bytes.Equal(NewEncoder(merges).Decode(back), []byte(text)) {
			t.Fatalf("binary=%v: decode after reload lost data", bin)
		}
		os.Remove(tp)
	}
}
