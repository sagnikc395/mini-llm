## tokenizer

an multi-threaded BPE Tokenizer implemented from scratch in Go
to parse and generate tokens at a very high speed.

Phase structure

1. Pre-tokenization and word counting. Split the corpus with a regex (GPT-2/GPT-4 style pattern), build a HashMap<Vec<u8>, u32> of pre-token to frequency. Embarrassingly parallel, and on a large corpus this is usually 60 to 80 percent of training wall time. This is where threading pays.

2. Merge learning. vocab_size - 256 iterations, each depending on the previous merge. Strictly sequential at the outer level.

3. Encoding. Embarrassingly parallel over documents, plus a cache that makes it mostly a hash lookup in steady state.

Phase 1: parallel counting

Memory-map the file. Compute nominal split points at i * len / n_threads, then advance each forward to the next safe delimiter so no pre-token straddles a boundary. A newline works for the standard regex patterns; a special token like <|endoftext|> is safer if your corpus has one.

Each worker builds a local counter. Reduce with a tree merge (pairwise, log depth) rather than every thread contending on one global map. Do not use a single Mutex<HashMap>; the lock will dominate. A sharded map keyed by hash prefix is the middle option if you want streaming.

Important consequence: after this phase your corpus is compressed to unique types, typically a few million entries for a multi-GB corpus. Phase 2 operates on types, weighted by count, never on raw text again.

Phase 2: the merge loop

The naive version recounts all adjacent pairs every iteration. That is O(vocab_size × corpus_types) and will take hours. You want incremental updates:

words:      Vec<Word>                        // each word is a token-id sequence
freqs:      Vec<u32>
pair_count: HashMap<(u32,u32), i64>
pair_index: HashMap<(u32,u32), HashSet<usize>>   // which words contain this pair
heap:       BinaryHeap<(count, pair)>             // lazy deletion

Each iteration:

Pop from the heap; if count != pair_count[pair], the entry is stale, discard and pop again.
For each word index in pair_index[best], rewrite that word's symbol sequence and emit deltas: decrement (prev, a) and (b, next), increment (prev, new) and (new, next), scaled by the word's frequency.
Apply deltas, push touched pairs back onto the heap, append best to the merge list.

Two details that cause silent bugs:

Overlapping occurrences. In a a a, merging (a,a) must consume positions 0-1 then skip to index 2, not merge at 1-2 as well. Advance by 2 after a hit.
Deterministic tie-breaking. When two pairs have equal count, pick by a fixed rule (lexicographically smallest pair id, say). Without this your vocab is not reproducible across runs, and you cannot diff against a reference implementation.

Use a doubly linked list of symbols per word (prev/next indices into a flat arena) if you want O(1) merges without shifting. A Vec with in-place compaction is simpler and fine for typical pre-token lengths under 20 bytes.

Where threading helps here: the initial full pair count over all words, and the step-2 loop when pair_index[best] is large. The affected set shrinks fast; after the first few hundred merges you are often touching only a few thousand words, and the delta-map reduction costs more than the work saved. Gate it: if affected.len() > THRESHOLD { parallel } else { serial }. Tune THRESHOLD empirically, it is usually in the low thousands.

Phase 3: encoding

Per pre-token, apply merges in rank order, not greedily left to right. The standard loop is: find the adjacent pair with the lowest merge rank, merge it, repeat until no pair is in the merge table. For pre-tokens under ~10 symbols a linear scan per round beats a heap because of constant factors.

Then add a cache: pre-token bytes -> Vec<u32>. Pre-token frequency is Zipfian, so hit rates above 95 percent are normal and the cache turns encoding into hashing plus memcpy. Use thread-local caches with a shared read-mostly map, or a sharded concurrent map. A single locked cache will negate your threading.

Parallelize over documents or chunks, again with boundary-safe splitting.

Gotchas that will bite
Regex engine. The GPT-4 pattern uses lookahead. Rust's regex crate does not support it; you need fancy-regex (slower) or a hand-written scanner. Python's re does not fully support it either for the \p{L} classes; regex (the PyPI package) does.
Merges never cross pre-token boundaries. This is what makes the vocab behave sanely and what lets you parallelize at all.
Special tokens. Split them out before the regex runs. BPE must never see them.
Byte-level base vocab. Start with all 256 byte values. This guarantees decode(encode(x)) == x for arbitrary bytes and eliminates UNK entirely. Do not start from characters.
Allocator contention. In a threaded build, a global malloc can serialize you. Try mimalloc or jemalloc before concluding your parallelism is broken.
Build order
Byte-level BPE, single thread, naive recount, 1 MB corpus. Assert round-trip on random bytes.
Add regex pre-tokenization. Load GPT-2's merges.txt and vocab.json and verify your encoder matches tiktoken token-for-token on a few MB of text. This validates phase 3 independently of phase 2.
Replace the naive recount with the incremental heap version. Assert the merge list is identical to step 1's on the same corpus.
Parallelize counting, then encoding, then the gated inner loop.

Measure throughput in MB of input text per second, not tokens per second; tokens/s conflates speed with compression ratio.

References
Sennrich, Haddow, Birch (2016), the original subword BPE paper: https://aclanthology.org/P16-1162/
Kudo and Richardson (2018), SentencePiece, useful for the pre-tokenization and normalization design decisions: https://aclanthology.org/D18-2012/

### problems with Go

Problem 1: Go's regexp cannot express the pre-tokenizer

regexp is RE2-backed, so no lookahead. The GPT-2 pattern needs \s+(?!\S). You have three options, and only one is good:

dlclark/regexp2 supports lookahead but backtracks and is roughly an order of magnitude slower. This will become your bottleneck.
Approximating the pattern in RE2 changes tokenization. Do not.
Hand-write the scanner. This is the correct answer and it is about 80 lines.

The alternation order in the regex is the scanner's branch order. At each position:

go
// GPT-2: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
func pretokenize(b []byte, emit func([]byte)) {
	i := 0
	for i < len(b) {
		// 1. contractions
		if b[i] == '\'' {
			if n := matchContraction(b[i:]); n > 0 {
				emit(b[i : i+n]); i += n; continue
			}
		}

		// 2-4. optional single space, then a run of one class
		j := i
		if b[j] == ' ' {
			j++
		}
		if j < len(b) {
			r, sz := utf8.DecodeRune(b[j:])
			switch {
			case unicode.IsLetter(r):
				k := j
				for k < len(b) {
					r, sz = utf8.DecodeRune(b[k:])
					if !unicode.IsLetter(r) { break }
					k += sz
				}
				emit(b[i:k]); i = k; continue
			case unicode.Is(unicode.N, r):
				// same shape, unicode.Is(unicode.N, r)
			case !unicode.IsSpace(r):
				// same shape, "other" class
			}
		}

		// 5. \s+(?!\S): consume the run, give back the last byte if
		//    a non-space follows
		k := i
		for k < len(b) {
			r, sz := utf8.DecodeRune(b[k:])
			if !unicode.IsSpace(r) { break }
			k += sz
		}
		if k < len(b) && k > i {
			_, lastSz := utf8.DecodeLastRune(b[i:k])
			k -= lastSz
		}
		if k == i { k++ } // defensive, should be unreachable
		emit(b[i:k]); i = k
	}
}

Note the optional-space branches must fall through to i unchanged if the class check fails, exactly like regex backtracking on " ?". Getting that wrong is the most common bug here.

One caveat: unicode.IsSpace in Go and \s in Python's regex package differ on a handful of codepoints. If you want byte-exact parity with tiktoken, fuzz your scanner against it on random Unicode and fix the divergences you find rather than trying to reason them out in advance.

Problem 2: GC pressure and map cost

Go's map is fine but not fast, and pre-tokenizing a multi-GB corpus into millions of small strings will put real load on the GC.

Pack pairs into a single uint64. map[uint64]int64 is substantially faster than a struct or array key:

go
func mkPair(a, b uint32) uint64 { return uint64(a)<<32 | uint64(b) }

Avoid allocating on map lookup. The compiler special-cases m[string(b)] in lookup position to skip the allocation, but m[string(b)]++ is an assignment and will allocate. During counting you cannot avoid one allocation per novel type, so structure it as: lookup with the no-alloc form, and only materialize the string when inserting.

Raise GOGC during training. debug.SetGCPercent(400) is usually a clear win for this workload. Check with GODEBUG=gctrace=1 before and after.

Use flat arenas for the trainer, not per-word slices. Pointer-free structures are much cheaper for the GC to scan:

go
type Trainer struct {
	symID   []uint32 // token id per slot
	symPrev []int32
	symNext []int32  // -1 sentinel; linked list inside the arena
	wordHead []int32 // index into sym arrays
	wordFreq []int32

	pairCount map[uint64]int64
	pairWords map[uint64][]int32 // word indices containing this pair
}

The linked list gives you O(1) merge application without shifting, and living inside three flat []int32/[]uint32 slices means zero pointers for the GC to trace.

Parallel counting
go
func splitBounds(data []byte, n int) []int {
	b := make([]int, n+1)
	b[n] = len(data)
	for i := 1; i < n; i++ {
		p := i * len(data) / n
		if k := bytes.IndexByte(data[p:], '\n'); k >= 0 {
			p += k + 1
		} else {
			p = len(data)
		}
		b[i] = p
	}
	return b
}

Then one goroutine per range, each building a local map[string]uint32, and a tree merge at the end (pairwise, log depth) rather than every goroutine contending on one map. Do not send tokens over a channel; channel ops are around 100ns and will dwarf the actual work.

For the file itself, golang.org/x/exp/mmap or a raw syscall.Mmap avoids copying the corpus into the heap.

Merge loop parallelism

The clean property that makes this safe: within one merge iteration, each affected word is rewritten by exactly one goroutine, and words are disjoint regions of the arena. So the symbol arena needs no synchronization. Only the pair-count deltas need merging, and those are per-goroutine map[uint64]int64 reduced serially after the barrier.

Gate it on size:

go
if len(affected) > parallelThreshold {
    // shard affected across workers, reduce deltas
} else {
    // serial
}

The affected set collapses fast after the first few hundred merges. Tune the threshold with a benchmark; it usually lands in the low thousands, below which goroutine spin-up plus delta reduction costs more than it saves.

For the priority queue, hand-roll a slice-based heap rather than using container/heap. The interface dispatch on Less/Swap is measurable in a loop this hot. Your less must break count ties on the packed pair value so the vocab is reproducible.

Encoder

Use a merges map keyed on the packed pair, not string concatenation:

go
merges  map[uint64]uint32 // mkPair(a,b) -> rank
merged  []uint32          // rank -> resulting token id

Per pre-token, repeatedly find the lowest-rank adjacent pair and apply it. Linear scan beats a heap at these lengths. Keep the working buffer per-goroutine and reuse it so encoding allocates nothing in steady state.

For the cache, start with a plain map[string][]uint32 per worker goroutine, no locking at all. Some duplication across workers, zero contention. Move to a 64-way sharded map with sync.RWMutex only if memory becomes the constraint. Do not reach for sync.Map; its access pattern is wrong for this.

Validation path

Load GPT-2's vocab.json and merges.txt and check your encoder against tiktoken on a few MB of mixed text before you trust your trainer at all. That isolates phase 3 from phase 2. Then train on a small corpus with both your naive and incremental implementations and assert the merge lists are byte-identical.

Round-trip property test with testing/quick over random []byte (not random strings, you want invalid UTF-8 in there too).

For profiling: go test -bench . -cpuprofile cpu.out, then go tool pprof -http=:8080. Also run with -race once on the parallel paths even though you have designed the disjointness in; it catches the case where your sharding has an off-by-one.


## References

1. https://tursunboev.com/posts/2024-08-20-bpe-in-golang/
