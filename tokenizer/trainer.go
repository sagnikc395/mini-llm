package main

import (
	"sync"
)

// Phase 2: the merge loop.
//
// The naive version recounts every adjacent pair on every iteration, which is
// O(vocab_size x corpus_types) and takes hours. This is the incremental
// version: a pair index says which words contain a pair, a lazy-deletion heap
// picks the best pair, and rewriting a word emits count deltas for only the
// four pairs its neighbourhood touches.
//
// The whole trainer lives in flat []uint32/[]int32 arenas. Per-word slices
// would be millions of pointers for the GC to trace on every cycle; these are
// pointer-free, so a scan costs nothing.

// Merge records one learned merge: (A, B) -> New, in rank order.
type Merge struct {
	A, B, New uint32
}

// Packing a pair into one uint64 makes map[uint64]int64 substantially faster
// than a struct or array key: no hashing of composite types, no equality
// function call.
func mkPair(a, b uint32) uint64 { return uint64(a)<<32 | uint64(b) }
func unPair(p uint64) (uint32, uint32) {
	return uint32(p >> 32), uint32(p & 0xffffffff)
}

const deadSym = ^uint32(0)

// delta accumulates pair-count changes and new pair->word entries produced by
// one worker within a single merge iteration.
type delta struct {
	counts map[uint64]int64
	words  map[uint64][]int32
}

func newDelta(hint int) *delta {
	return &delta{
		counts: make(map[uint64]int64, hint),
		words:  make(map[uint64][]int32, hint),
	}
}

func (d *delta) mergeFrom(o *delta) {
	for p, v := range o.counts {
		d.counts[p] += v
	}
	for p, ws := range o.words {
		d.words[p] = append(d.words[p], ws...)
	}
}

func treeMergeDeltas(ds []*delta) *delta {
	for len(ds) > 1 {
		var wg sync.WaitGroup
		next := ds[:0:0]
		for i := 0; i+1 < len(ds); i += 2 {
			a, b := ds[i], ds[i+1]
			next = append(next, a)
			wg.Add(1)
			go func() {
				defer wg.Done()
				a.mergeFrom(b)
			}()
		}
		if len(ds)%2 == 1 {
			next = append(next, ds[len(ds)-1])
		}
		wg.Wait()
		ds = next
	}
	if len(ds) == 0 {
		return newDelta(0)
	}
	return ds[0]
}

type Trainer struct {
	// Symbol arena. One slot per byte of every unique type; words are
	// contiguous ranges linked by prev/next so a merge is O(1) with no
	// shifting. -1 is the list sentinel.
	symID   []uint32
	symPrev []int32
	symNext []int32

	wordHead []int32
	wordFreq []int64

	pairCount map[uint64]int64
	pairWords map[uint64][]int32 // word indices containing this pair
	heap      pairHeap

	// stamp/epoch dedupe the affected-word list in O(n) without a set.
	stamp []int32
	epoch int32

	// Parallelism knobs.
	workers   int
	threshold int
}

// NewTrainer lays the counted types out in the arena. toks[i] is a pre-token,
// freqs[i] its corpus frequency; phase 2 works on types weighted by count and
// never sees raw text again.
func NewTrainer(toks []string, freqs []int64, workers, threshold int) *Trainer {
	total := 0
	for _, s := range toks {
		total += len(s)
	}
	t := &Trainer{
		symID:     make([]uint32, 0, total),
		symPrev:   make([]int32, 0, total),
		symNext:   make([]int32, 0, total),
		wordHead:  make([]int32, 0, len(toks)),
		wordFreq:  make([]int64, 0, len(toks)),
		stamp:     make([]int32, 0, len(toks)),
		workers:   workers,
		threshold: threshold,
	}
	for i, s := range toks {
		if len(s) == 0 {
			continue
		}
		head := int32(len(t.symID))
		for j := 0; j < len(s); j++ {
			idx := int32(len(t.symID))
			t.symID = append(t.symID, uint32(s[j])) // byte-level base vocab
			if j == 0 {
				t.symPrev = append(t.symPrev, -1)
			} else {
				t.symPrev = append(t.symPrev, idx-1)
			}
			if j == len(s)-1 {
				t.symNext = append(t.symNext, -1)
			} else {
				t.symNext = append(t.symNext, idx+1)
			}
		}
		t.wordHead = append(t.wordHead, head)
		t.wordFreq = append(t.wordFreq, freqs[i])
		t.stamp = append(t.stamp, -1)
	}
	t.initialCount()
	return t
}

// initialCount builds the full pair count and pair index. This is the one part
// of phase 2 that is trivially parallel over all words.
func (t *Trainer) initialCount() {
	nw := len(t.wordHead)
	w := t.workers
	if w > 1 && nw >= 4096 {
		parts := make([]*delta, w)
		var wg sync.WaitGroup
		for k := 0; k < w; k++ {
			lo, hi := k*nw/w, (k+1)*nw/w
			d := newDelta(1 << 12)
			parts[k] = d
			wg.Add(1)
			go func() {
				defer wg.Done()
				t.countRange(lo, hi, d)
			}()
		}
		wg.Wait()
		d := treeMergeDeltas(parts)
		t.pairCount, t.pairWords = d.counts, d.words
	} else {
		d := newDelta(1 << 12)
		t.countRange(0, nw, d)
		t.pairCount, t.pairWords = d.counts, d.words
	}

	t.heap.a = make([]heapEntry, 0, len(t.pairCount))
	for p, c := range t.pairCount {
		t.heap.Push(c, p)
	}
}

func (t *Trainer) countRange(lo, hi int, d *delta) {
	for wi := lo; wi < hi; wi++ {
		f := t.wordFreq[wi]
		for p := t.wordHead[wi]; p >= 0; p = t.symNext[p] {
			q := t.symNext[p]
			if q < 0 {
				break
			}
			pr := mkPair(t.symID[p], t.symID[q])
			d.counts[pr] += f
			if ws := d.words[pr]; len(ws) == 0 || ws[len(ws)-1] != int32(wi) {
				d.words[pr] = append(ws, int32(wi))
			}
		}
	}
}

// popBest returns the highest-count pair, discarding stale heap entries. Every
// count change pushes a fresh entry, so the current maximum is always present.
func (t *Trainer) popBest() (uint64, int64, bool) {
	for {
		e, ok := t.heap.Pop()
		if !ok {
			return 0, 0, false
		}
		if c, live := t.pairCount[e.pair]; live && c == e.count && c > 0 {
			return e.pair, e.count, true
		}
	}
}

// affectedWords takes ownership of the pair's word list and dedupes it in
// place; entries can repeat because deltas append without checking.
func (t *Trainer) affectedWords(pair uint64) []int32 {
	list := t.pairWords[pair]
	delete(t.pairWords, pair)
	t.epoch++
	out := list[:0]
	for _, w := range list {
		if t.stamp[w] != t.epoch {
			t.stamp[w] = t.epoch
			out = append(out, w)
		}
	}
	return out
}

// applyWord rewrites one word's symbol sequence, replacing every occurrence of
// (a,b) with nid and emitting the surrounding count deltas.
func (t *Trainer) applyWord(wi int32, a, b, nid uint32, d *delta) {
	f := t.wordFreq[wi]
	ab := mkPair(a, b)
	for p := t.wordHead[wi]; p >= 0; {
		q := t.symNext[p]
		if q < 0 {
			return
		}
		if t.symID[p] != a || t.symID[q] != b {
			p = q
			continue
		}

		prev, next := t.symPrev[p], t.symNext[q]
		if prev >= 0 {
			d.counts[mkPair(t.symID[prev], a)] -= f
			np := mkPair(t.symID[prev], nid)
			d.counts[np] += f
			d.words[np] = append(d.words[np], wi)
		}
		if next >= 0 {
			d.counts[mkPair(b, t.symID[next])] -= f
			np := mkPair(nid, t.symID[next])
			d.counts[np] += f
			d.words[np] = append(d.words[np], wi)
		}
		d.counts[ab] -= f

		// splice q out of the list; p becomes the merged symbol.
		t.symID[p] = nid
		t.symNext[p] = next
		if next >= 0 {
			t.symPrev[next] = p
		}
		t.symID[q] = deadSym

		// Resume *after* the merged symbol. In "a a a", merging (a,a) must
		// consume positions 0-1 and then skip to 2, never merging 1-2 as well.
		p = next
	}
}

func (t *Trainer) applyParallel(affected []int32, a, b, nid uint32) *delta {
	w := t.workers
	if w > len(affected) {
		w = len(affected)
	}
	parts := make([]*delta, w)
	var wg sync.WaitGroup
	for k := 0; k < w; k++ {
		lo, hi := k*len(affected)/w, (k+1)*len(affected)/w
		d := newDelta(64)
		parts[k] = d
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Safe without synchronisation: within one iteration each
			// affected word is rewritten by exactly one goroutine, and words
			// are disjoint regions of the arena. Only the deltas need merging.
			for _, wi := range affected[lo:hi] {
				t.applyWord(wi, a, b, nid, d)
			}
		}()
	}
	wg.Wait()
	return treeMergeDeltas(parts)
}

func (t *Trainer) applyDelta(d *delta) {
	for p, dv := range d.counts {
		if dv == 0 {
			continue
		}
		c := t.pairCount[p] + dv
		if c <= 0 {
			delete(t.pairCount, p)
			delete(t.pairWords, p)
			continue
		}
		t.pairCount[p] = c
		t.heap.Push(c, p)
	}
	for p, ws := range d.words {
		if _, live := t.pairCount[p]; live {
			t.pairWords[p] = append(t.pairWords[p], ws...)
		}
	}
}

// Train runs numMerges iterations. The outer loop is strictly sequential —
// each merge depends on the previous one — so threading only pays inside an
// iteration, and only while the affected set is still large.
func (t *Trainer) Train(numMerges int, progress func(rank int, m Merge, count int64)) []Merge {
	merges := make([]Merge, 0, numMerges)
	next := uint32(256)
	for len(merges) < numMerges {
		pair, count, ok := t.popBest()
		if !ok {
			break
		}
		a, b := unPair(pair)
		nid := next
		next++

		affected := t.affectedWords(pair)
		var d *delta
		// The affected set collapses fast; after a few hundred merges the
		// delta reduction costs more than the work saved, so gate on size.
		if t.workers > 1 && len(affected) > t.threshold {
			d = t.applyParallel(affected, a, b, nid)
		} else {
			d = newDelta(64)
			for _, wi := range affected {
				t.applyWord(wi, a, b, nid, d)
			}
		}
		t.applyDelta(d)
		delete(t.pairCount, pair)
		delete(t.pairWords, pair)

		m := Merge{a, b, nid}
		merges = append(merges, m)
		if progress != nil {
			progress(len(merges), m, count)
		}
	}
	return merges
}

// BuildVocab expands the merge list into id -> bytes. Ids 0-255 are the raw
// bytes, which is what guarantees decode(encode(x)) == x for arbitrary input
// and eliminates UNK entirely.
func BuildVocab(merges []Merge) [][]byte {
	vocab := make([][]byte, 256+len(merges))
	for i := 0; i < 256; i++ {
		vocab[i] = []byte{byte(i)}
	}
	for _, m := range merges {
		v := make([]byte, 0, len(vocab[m.A])+len(vocab[m.B]))
		v = append(v, vocab[m.A]...)
		v = append(v, vocab[m.B]...)
		vocab[m.New] = v
	}
	return vocab
}
