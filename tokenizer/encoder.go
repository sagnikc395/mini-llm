package main

import (
	"runtime"
	"sync"
)

// Phase 3: encoding. Embarrassingly parallel over chunks, and in steady state
// mostly a hash lookup thanks to the pre-token cache.

type Encoder struct {
	ranks  map[uint64]uint32 // packed pair -> merge rank
	newID  []uint32          // rank -> resulting token id
	vocab  [][]byte          // id -> bytes, for decoding
	merges []Merge
}

func NewEncoder(merges []Merge) *Encoder {
	e := &Encoder{
		ranks:  make(map[uint64]uint32, len(merges)),
		newID:  make([]uint32, len(merges)),
		vocab:  BuildVocab(merges),
		merges: merges,
	}
	for rank, m := range merges {
		e.ranks[mkPair(m.A, m.B)] = uint32(rank)
		e.newID[rank] = m.New
	}
	return e
}

func (e *Encoder) VocabSize() int { return len(e.vocab) }

// encodeToken applies merges in rank order, not greedily left to right: find
// the adjacent pair with the lowest rank, merge it, repeat. For pre-tokens
// under ~10 symbols a linear scan per round beats a heap on constant factors.
//
// buf is caller-owned and reused, so steady-state encoding allocates nothing.
func (e *Encoder) encodeToken(tok []byte, buf []uint32) []uint32 {
	buf = buf[:0]
	for _, c := range tok {
		buf = append(buf, uint32(c))
	}
	for len(buf) >= 2 {
		bestRank := ^uint32(0)
		bestI := -1
		for i := 0; i+1 < len(buf); i++ {
			if r, ok := e.ranks[mkPair(buf[i], buf[i+1])]; ok && r < bestRank {
				bestRank, bestI = r, i
			}
		}
		if bestI < 0 {
			break
		}
		buf[bestI] = e.newID[bestRank]
		buf = append(buf[:bestI+1], buf[bestI+2:]...)
	}
	return buf
}

// encodeChunk encodes a boundary-safe byte range using a private cache. Plain
// map, no locking: some duplication across workers, zero contention. A single
// locked cache would negate the threading, and sync.Map's access pattern is
// wrong for this.
func (e *Encoder) encodeChunk(data []byte, out []uint32) []uint32 {
	cache := make(map[string][]uint32, 1<<12)
	buf := make([]uint32, 0, 64)
	s := scanner{b: data}
	for {
		lo, hi, ok := s.next()
		if !ok {
			return out
		}
		tok := data[lo:hi]
		// Pre-token frequency is Zipfian, so hit rates above 95% are normal
		// and this turns encoding into hashing plus memcpy.
		if ids, hit := cache[string(tok)]; hit {
			out = append(out, ids...)
			continue
		}
		buf = e.encodeToken(tok, buf)
		ids := make([]uint32, len(buf))
		copy(ids, buf)
		cache[string(tok)] = ids
		out = append(out, ids...)
	}
}

// Encode tokenizes data with `workers` goroutines over boundary-safe chunks
// and concatenates the per-chunk results in order.
func (e *Encoder) Encode(data []byte, workers int) []uint32 {
	if workers < 1 {
		workers = runtime.NumCPU()
	}
	if len(data) == 0 {
		return nil
	}
	if workers == 1 {
		return e.encodeChunk(data, make([]uint32, 0, len(data)/3))
	}

	bounds := splitBounds(data, workers)
	parts := make([][]uint32, workers)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		lo, hi := bounds[i], bounds[i+1]
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			parts[i] = e.encodeChunk(data[lo:hi], make([]uint32, 0, (hi-lo)/3+8))
		}()
	}
	wg.Wait()

	n := 0
	for _, p := range parts {
		n += len(p)
	}
	out := make([]uint32, 0, n)
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

// Decode is the inverse: concatenate each id's byte expansion.
func (e *Encoder) Decode(ids []uint32) []byte {
	n := 0
	for _, id := range ids {
		if int(id) < len(e.vocab) {
			n += len(e.vocab[id])
		}
	}
	out := make([]byte, 0, n)
	for _, id := range ids {
		if int(id) < len(e.vocab) {
			out = append(out, e.vocab[id]...)
		}
	}
	return out
}
