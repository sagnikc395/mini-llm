package main

import (
	"math/rand"
	"strings"
)

// randomText builds Zipfian-ish text from a small word pool with mixed
// punctuation, whitespace and multi-byte runes — close enough to a real corpus
// that the merge loop has something to chew on.
var wordPool = []string{
	"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
	"token", "tokenizer", "merge", "merges", "pair", "pairs", "byte",
	"encode", "decode", "corpus", "naïve", "café", "résumé", "日本語",
	"don't", "it's", "we'll", "they've", "0", "42", "1234", "½",
}

func randomText(r *rand.Rand, n int) string {
	var b strings.Builder
	for b.Len() < n {
		// Zipf-ish: bias hard toward the head of the pool.
		i := int(float64(len(wordPool)) * r.Float64() * r.Float64())
		if i >= len(wordPool) {
			i = len(wordPool) - 1
		}
		b.WriteString(wordPool[i])
		switch r.Intn(10) {
		case 0:
			b.WriteString(".\n")
		case 1:
			b.WriteString(", ")
		case 2:
			b.WriteString("  ")
		default:
			b.WriteString(" ")
		}
	}
	return b.String()
}
