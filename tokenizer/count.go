package main

import (
	"bytes"
	"runtime"
	"sync"
)

// Phase 1: pre-tokenization and type counting.
//
// This is embarrassingly parallel and, on a large corpus, 60-80% of training
// wall time. Each worker owns a boundary-safe byte range and builds a local
// counter; the locals are reduced with a pairwise tree merge. A single
// Mutex[map] here would be dominated by lock contention, and shipping tokens
// over a channel would cost ~100ns per token against maybe 20ns of real work.

// counter maps a pre-token to a slot in a flat counts slice.
//
// The indirection exists to dodge an allocation: the compiler special-cases
// m[string(b)] in *lookup* position to avoid materializing the string, but
// m[string(b)]++ is an assignment and allocates every time. Looking up an
// index and bumping counts[i] keeps the steady-state path allocation-free;
// only a novel type pays for a string.
type counter struct {
	idx    map[string]int32
	toks   []string
	counts []int64
}

func newCounter(hint int) *counter {
	return &counter{idx: make(map[string]int32, hint)}
}

func (c *counter) add(tok []byte) {
	if i, ok := c.idx[string(tok)]; ok { // no-alloc lookup form
		c.counts[i]++
		return
	}
	s := string(tok)
	c.idx[s] = int32(len(c.counts))
	c.toks = append(c.toks, s)
	c.counts = append(c.counts, 1)
}

func (c *counter) addN(tok string, n int64) {
	if i, ok := c.idx[tok]; ok {
		c.counts[i] += n
		return
	}
	c.idx[tok] = int32(len(c.counts))
	c.toks = append(c.toks, tok)
	c.counts = append(c.counts, n)
}

func (c *counter) mergeFrom(o *counter) {
	for i, t := range o.toks {
		c.addN(t, o.counts[i])
	}
}

// splitBounds picks n nominal split points and advances each forward to the
// byte after the next newline, so no pre-token straddles a boundary. A newline
// is safe for the standard patterns: it is always whitespace, and merges never
// cross pre-token boundaries.
func splitBounds(data []byte, n int) []int {
	if n < 1 {
		n = 1
	}
	b := make([]int, n+1)
	b[n] = len(data)
	for i := 1; i < n; i++ {
		p := i * len(data) / n
		if p < b[i-1] {
			p = b[i-1]
		}
		if k := bytes.IndexByte(data[p:], '\n'); k >= 0 {
			p += k + 1
		} else {
			p = len(data)
		}
		b[i] = p
	}
	return b
}

// CountTypes pre-tokenizes data in parallel and returns unique pre-tokens with
// their frequencies. After this phase the corpus is compressed to types —
// typically a few million entries even for a multi-GB corpus — and phase 2
// never touches raw text again.
func CountTypes(data []byte, workers int) (toks []string, freqs []int64) {
	if workers < 1 {
		workers = runtime.NumCPU()
	}
	if len(data) == 0 {
		return nil, nil
	}

	bounds := splitBounds(data, workers)
	locals := make([]*counter, 0, workers)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		lo, hi := bounds[i], bounds[i+1]
		if lo >= hi {
			continue
		}
		c := newCounter(1 << 12)
		locals = append(locals, c)
		wg.Add(1)
		go func(c *counter, chunk []byte) {
			defer wg.Done()
			s := scanner{b: chunk}
			for {
				lo, hi, ok := s.next()
				if !ok {
					return
				}
				c.add(chunk[lo:hi])
			}
		}(c, data[lo:hi])
	}
	wg.Wait()

	c := treeMergeCounters(locals)
	if c == nil {
		return nil, nil
	}
	return c.toks, c.counts
}

// treeMergeCounters reduces pairwise at log depth instead of funnelling every
// worker into one map.
func treeMergeCounters(cs []*counter) *counter {
	for len(cs) > 1 {
		var wg sync.WaitGroup
		next := cs[:0:0]
		for i := 0; i+1 < len(cs); i += 2 {
			a, b := cs[i], cs[i+1]
			next = append(next, a)
			wg.Add(1)
			go func() {
				defer wg.Done()
				a.mergeFrom(b)
			}()
		}
		if len(cs)%2 == 1 {
			next = append(next, cs[len(cs)-1])
		}
		wg.Wait()
		cs = next
	}
	if len(cs) == 0 {
		return nil
	}
	return cs[0]
}
