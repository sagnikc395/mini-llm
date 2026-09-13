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

// tokenCache maps a pre-token to a span of ids in a flat arena.
//
// The arena exists to kill an allocation: a map[string][]uint32 costs a slice
// header plus a backing array per entry, and with tens of thousands of types
// per worker that allocation traffic shows up as GC and madvise time. Offsets
// stay valid when the arena grows, so lookups are a hash plus a reslice.
type tokenCache struct {
	idx   map[string]uint64 // key -> off<<32 | length
	arena []uint32
}

func newTokenCache() *tokenCache {
	return &tokenCache{idx: make(map[string]uint64, 1<<12), arena: make([]uint32, 0, 1<<16)}
}

// encodeChunk encodes a boundary-safe byte range using a private cache. Plain
// map, no locking: some duplication across workers, zero contention. A single
// locked cache would negate the threading, and sync.Map's access pattern is
// wrong for this.
func (e *Encoder) encodeChunk(data []byte, out []uint32) []uint32 {
	c := newTokenCache()
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
		if v, hit := c.idx[string(tok)]; hit { // no-alloc lookup form
			off, n := uint32(v>>32), uint32(v)
			out = append(out, c.arena[off:off+n]...)
			continue
		}
		buf = e.encodeToken(tok, buf)
		off := uint32(len(c.arena))
		c.arena = append(c.arena, buf...)
		c.idx[string(tok)] = uint64(off)<<32 | uint64(len(buf))
		out = append(out, buf...)
	}
}

// EncodeChunks tokenizes data in parallel and returns the per-chunk id slices
// in order. Callers that stream the result (the CLI writes straight to a file)
// should prefer this over Encode: concatenating the parts costs a full copy of
// the token stream, which for a large corpus is tens of MB of pure memmove.
func (e *Encoder) EncodeChunks(data []byte, workers int) [][]uint32 {
	if workers < 1 {
		workers = runtime.NumCPU()
	}
	if len(data) == 0 {
		return nil
	}
	if workers == 1 {
		return [][]uint32{e.encodeChunk(data, make([]uint32, 0, estimateTokens(len(data))))}
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
			parts[i] = e.encodeChunk(data[lo:hi], make([]uint32, 0, estimateTokens(hi-lo)))
		}()
	}
	wg.Wait()
	return parts
}

// estimateTokens pre-sizes an output buffer. Byte-level BPE at a few thousand
// merges lands around 4 bytes per token on real text; guessing a little high
// wastes memory, guessing low costs a regrow and a copy.
func estimateTokens(nbytes int) int { return nbytes/4 + 16 }

// Encode tokenizes data with `workers` goroutines over boundary-safe chunks
// and concatenates the per-chunk results in order.
func (e *Encoder) Encode(data []byte, workers int) []uint32 {
	parts := e.EncodeChunks(data, workers)
	if len(parts) == 1 {
		return parts[0]
	}
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
