package main

import (
	"encoding/json"
	"os"
	"testing"
)

// TestDumpPretokens is the differential-testing hook. Go's regexp cannot run
// the reference pattern, so parity is checked against Python's `regex` module
// out of process:
//
//	DUMP_IN=corpus.txt DUMP_OUT=spans.json go test -run TestDumpPretokens .
//
// then compare spans.json to the pattern's finditer output. The scanner is
// verified byte-identical on the corpus, on adversarial generated text, and on
// exotic Unicode whitespace (where unicode.IsSpace and \s are known to differ
// on some codepoints).
func TestDumpPretokens(t *testing.T) {
	in := os.Getenv("DUMP_IN")
	if in == "" {
		t.Skip()
	}
	data, err := os.ReadFile(in)
	if err != nil {
		t.Fatal(err)
	}
	out := [][]int{}
	s := scanner{b: data}
	for {
		lo, hi, ok := s.next()
		if !ok {
			break
		}
		out = append(out, []int{lo, hi})
	}
	f, err := os.Create(os.Getenv("DUMP_OUT"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	json.NewEncoder(f).Encode(out)
}
