package main

import (
	"bytes"
	"math/rand"
	"testing"
	"testing/quick"
)

func pretokens(s string) []string {
	var out []string
	Pretokenize([]byte(s), func(b []byte) { out = append(out, string(b)) })
	return out
}

func TestPretokenizeCases(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"", nil},
		{"hello world", []string{"hello", " world"}},
		{"Hello  world", []string{"Hello", " ", " world"}},
		{"don't", []string{"don", "'t"}},
		{"I'll've", []string{"I", "'ll", "'ve"}},
		{"it's a 'quoted' word", []string{"it", "'s", " a", " '", "quoted", "'", " word"}},
		{"abc123", []string{"abc", "123"}},
		{"a  b", []string{"a", " ", " b"}},
		{"   ", []string{"   "}},             // trailing run: plain \s+
		{"   a", []string{"  ", " a"}},       // \s+(?!\S) gives back the last space
		{"\n\nx", []string{"\n", "\n", "x"}}, // the " ?" prefix is a literal space, not \s
		{"hi!?", []string{"hi", "!?"}},
		{" !!", []string{" !!"}},
		{"naïve café", []string{"naïve", " café"}}, // multi-byte letters
		{"½ ½", []string{"½", " ½"}},               // \p{N} is No, not just Nd
		{"a\tb", []string{"a", "\t", "b"}},
		{"'RE'S", []string{"'", "RE", "'", "S"}}, // contractions are lowercase only
		{"2024-01-02", []string{"2024", "-", "01", "-", "02"}},
	}
	for _, c := range cases {
		got := pretokens(c.in)
		if len(got) != len(c.want) {
			t.Errorf("%q: got %q, want %q", c.in, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("%q: got %q, want %q", c.in, got, c.want)
				break
			}
		}
	}
}

// The scanner must tile its input: no gaps, no overlaps, no zero-length spans.
// This is what makes merges-never-cross-boundaries safe to parallelize and
// what makes the round trip hold for arbitrary bytes.
func TestPretokenizeTiles(t *testing.T) {
	f := func(b []byte) bool {
		s := scanner{b: b}
		pos := 0
		for {
			lo, hi, ok := s.next()
			if !ok {
				break
			}
			if lo != pos || hi <= lo || hi > len(b) {
				return false
			}
			pos = hi
		}
		return pos == len(b)
	}
	if err := quick.Check(f, &quick.Config{MaxCount: 2000}); err != nil {
		t.Fatal(err)
	}
	// Explicitly cover valid UTF-8 too; random []byte is mostly invalid.
	for i := 0; i < 200; i++ {
		s := randomText(rand.New(rand.NewSource(int64(i))), 512)
		if !f([]byte(s)) {
			t.Fatalf("tiling failed on %q", s)
		}
	}
}

func TestSplitBoundsPartitions(t *testing.T) {
	data := []byte("alpha\nbeta\ngamma\ndelta\nepsilon\nzeta\n")
	for n := 1; n <= 8; n++ {
		b := splitBounds(data, n)
		if b[0] != 0 || b[n] != len(data) {
			t.Fatalf("n=%d: bad endpoints %v", n, b)
		}
		var joined []byte
		for i := 0; i < n; i++ {
			if b[i] > b[i+1] {
				t.Fatalf("n=%d: non-monotonic %v", n, b)
			}
			if b[i] > 0 && b[i] < len(data) && data[b[i]-1] != '\n' {
				t.Fatalf("n=%d: boundary %d not after a newline", n, b[i])
			}
			joined = append(joined, data[b[i]:b[i+1]]...)
		}
		if !bytes.Equal(joined, data) {
			t.Fatalf("n=%d: chunks do not reassemble", n)
		}
	}
}
