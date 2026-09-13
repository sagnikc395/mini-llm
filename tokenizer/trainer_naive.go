package main

// A deliberately naive reference trainer: full recount of every adjacent pair
// on every iteration, over the same type-weighted word list. It is O(vocab x
// types) and far too slow for real corpora, but it is obviously correct, so
// the incremental trainer is tested for a byte-identical merge list against
// it. Same tie-break (highest count, then smallest packed pair) so the two are
// comparable.
func trainNaive(toks []string, freqs []int64, numMerges int) []Merge {
	words := make([][]uint32, 0, len(toks))
	ws := make([]int64, 0, len(toks))
	for i, s := range toks {
		if len(s) == 0 {
			continue
		}
		w := make([]uint32, len(s))
		for j := 0; j < len(s); j++ {
			w[j] = uint32(s[j])
		}
		words = append(words, w)
		ws = append(ws, freqs[i])
	}

	merges := make([]Merge, 0, numMerges)
	next := uint32(256)
	for len(merges) < numMerges {
		counts := make(map[uint64]int64)
		for i, w := range words {
			for j := 0; j+1 < len(w); j++ {
				counts[mkPair(w[j], w[j+1])] += ws[i]
			}
		}
		if len(counts) == 0 {
			break
		}
		var best uint64
		var bestC int64
		for p, c := range counts {
			if c > bestC || (c == bestC && p < best) {
				best, bestC = p, c
			}
		}
		if bestC <= 0 {
			break
		}
		a, b := unPair(best)
		nid := next
		next++
		for i, w := range words {
			out := w[:0]
			j := 0
			for j < len(w) {
				if j+1 < len(w) && w[j] == a && w[j+1] == b {
					out = append(out, nid)
					j += 2 // advance by 2: overlapping occurrences must not both merge
				} else {
					out = append(out, w[j])
					j++
				}
			}
			words[i] = out
		}
		merges = append(merges, Merge{a, b, nid})
	}
	return merges
}
