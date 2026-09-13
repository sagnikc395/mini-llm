package main

import (
	"unicode"
	"unicode/utf8"
)

// Hand-written scanner for the GPT-2 pre-tokenizer pattern:
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// Go's regexp is RE2-backed and cannot express the \s+(?!\S) lookahead, and
// regexp2's backtracking engine would become the bottleneck. The alternation
// order above is exactly the branch order below.
//
// The scanner is total: the emitted spans tile the input with no gaps and no
// overlaps, which is what makes decode(encode(x)) == x hold for arbitrary
// bytes, including invalid UTF-8 (a bad byte decodes to RuneError with size 1
// and lands in the "other" class).

const (
	clsSpace = iota
	clsLetter
	clsNumber
	clsOther
)

var asciiClass [utf8.RuneSelf]uint8

func init() {
	for i := range asciiClass {
		c := byte(i)
		switch {
		case c == ' ' || (c >= 0x09 && c <= 0x0d):
			asciiClass[i] = clsSpace
		case (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'):
			asciiClass[i] = clsLetter
		case c >= '0' && c <= '9':
			asciiClass[i] = clsNumber
		default:
			asciiClass[i] = clsOther
		}
	}
}

// classAt returns the character class at b[i] and the rune's encoded size.
// ASCII takes a table lookup; only the tail decodes a rune.
func classAt(b []byte, i int) (uint8, int) {
	if c := b[i]; c < utf8.RuneSelf {
		return asciiClass[c], 1
	}
	r, sz := utf8.DecodeRune(b[i:])
	switch {
	case unicode.IsLetter(r):
		return clsLetter, sz
	case unicode.Is(unicode.N, r): // \p{N}, not just Nd
		return clsNumber, sz
	case unicode.IsSpace(r):
		return clsSpace, sz
	}
	return clsOther, sz
}

// matchContraction returns the length of a leading 's|'t|'re|'ve|'m|'ll|'d,
// or 0. Lowercase only, matching the GPT-2 pattern.
func matchContraction(b []byte) int {
	if len(b) < 2 {
		return 0
	}
	switch b[1] {
	case 's', 't', 'm', 'd':
		return 2
	case 'r', 'v':
		if len(b) >= 3 && b[2] == 'e' {
			return 3
		}
	case 'l':
		if len(b) >= 3 && b[2] == 'l' {
			return 3
		}
	}
	return 0
}

// scanner yields pre-token spans. It is a struct rather than a callback-driven
// loop so the hot paths (counting, encoding) avoid a closure call per token.
type scanner struct {
	b []byte
	i int
}

// next returns the next pre-token as a [lo, hi) span into s.b.
func (s *scanner) next() (lo, hi int, ok bool) {
	b, n := s.b, len(s.b)
	i := s.i
	if i >= n {
		return 0, 0, false
	}

	// 1. contractions
	if b[i] == '\'' {
		if m := matchContraction(b[i:]); m > 0 {
			s.i = i + m
			return i, s.i, true
		}
	}

	// 2-4. an optional single space, then a run of one class. If the class
	// check fails we must fall through with i unchanged, exactly like regex
	// backtracking on " ?" — getting that wrong is the classic bug here.
	j := i
	if b[j] == ' ' {
		j++
	}
	if j < n {
		cls, sz := classAt(b, j)
		if cls != clsSpace {
			k := j + sz
			for k < n {
				c2, s2 := classAt(b, k)
				if c2 != cls {
					break
				}
				k += s2
			}
			s.i = k
			return i, k, true
		}
	}

	// 5. \s+(?!\S): consume the whitespace run, then give back the last rune
	// if a non-space follows. If that would leave an empty match the lookahead
	// branch fails outright and plain \s+ takes the whole run.
	e := i
	for e < n {
		c, sz := classAt(b, e)
		if c != clsSpace {
			break
		}
		e += sz
	}
	k := e
	if e < n {
		_, last := utf8.DecodeLastRune(b[i:e])
		if e-last > i {
			k = e - last
		}
	}
	if k == i {
		k = e
	}
	if k == i { // unreachable: every byte belongs to some class
		k = i + 1
	}
	s.i = k
	return i, k, true
}

// Pretokenize is the callback form, kept for tests and readability.
func Pretokenize(b []byte, emit func([]byte)) {
	s := scanner{b: b}
	for {
		lo, hi, ok := s.next()
		if !ok {
			return
		}
		emit(b[lo:hi])
	}
}
