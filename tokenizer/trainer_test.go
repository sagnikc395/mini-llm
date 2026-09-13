package main

import (
	"math/rand"
	"reflect"
	"testing"
)

func corpusTypes(t *testing.T, text string, workers int) ([]string, []int64) {
	t.Helper()
	toks, freqs := CountTypes([]byte(text), workers)
	if len(toks) != len(freqs) {
		t.Fatalf("counter shape mismatch")
	}
	return toks, freqs
}

// Counting must be independent of how the corpus is sharded.
func TestCountTypesParallelMatchesSerial(t *testing.T) {
	text := randomText(rand.New(rand.NewSource(7)), 300_000)
	want := map[string]int64{}
	toks, freqs := corpusTypes(t, text, 1)
	for i, s := range toks {
		want[s] = freqs[i]
	}
	// Sanity: the serial count must agree with a dead-simple recount.
	ref := map[string]int64{}
	Pretokenize([]byte(text), func(b []byte) { ref[string(b)]++ })
	if !reflect.DeepEqual(ref, want) {
		t.Fatalf("serial count disagrees with reference recount")
	}
	for _, w := range []int{2, 3, 8, 16} {
		got := map[string]int64{}
		toks, freqs := corpusTypes(t, text, w)
		for i, s := range toks {
			got[s] = freqs[i]
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("workers=%d: counts differ from serial", w)
		}
	}
}

// The whole point of the incremental trainer is that it produces exactly what
// the naive full-recount trainer would, only faster. Assert the merge lists are
// identical, at every thread count and on both sides of the parallel gate.
func TestIncrementalMatchesNaive(t *testing.T) {
	text := randomText(rand.New(rand.NewSource(11)), 200_000)
	toks, freqs := corpusTypes(t, text, 4)

	// Ask for more merges than the corpus can supply so both trainers also
	// have to agree on where to stop.
	const numMerges = 500
	want := trainNaive(toks, freqs, numMerges)
	if len(want) < 100 {
		t.Fatalf("corpus too small to be a meaningful test: %d merges", len(want))
	}

	for _, workers := range []int{1, 2, 8} {
		for _, threshold := range []int{0, 64, 1 << 30} { // always parallel .. never
			tr := NewTrainer(toks, freqs, workers, threshold)
			got := tr.Train(numMerges, nil)
			if !reflect.DeepEqual(got, want) {
				for i := range got {
					if i >= len(want) || got[i] != want[i] {
						t.Fatalf("workers=%d threshold=%d: merge %d = %v, naive says %v",
							workers, threshold, i, got[i], want[i])
					}
				}
				t.Fatalf("workers=%d threshold=%d: merge lists differ in length", workers, threshold)
			}
		}
	}
}

// Overlapping occurrences: in "aaaa", merging (a,a) must consume 0-1 then skip
// to 2, never merging 1-2 as well.
func TestOverlappingOccurrences(t *testing.T) {
	toks := []string{"aaaa", "aaa"}
	freqs := []int64{1, 1}
	tr := NewTrainer(toks, freqs, 1, 1<<30)
	merges := tr.Train(1, nil)
	if len(merges) != 1 {
		t.Fatalf("got %d merges", len(merges))
	}
	m := merges[0]
	if m.A != 'a' || m.B != 'a' {
		t.Fatalf("first merge = %v, want (a,a)", m)
	}
	// "aaaa" -> [aa][aa]; "aaa" -> [aa]a. So (256,256) has count 1 and
	// (256,'a') has count 1; a naive double-merge would corrupt these.
	if got := tr.pairCount[mkPair(256, 256)]; got != 1 {
		t.Errorf("count(256,256) = %d, want 1", got)
	}
	if got := tr.pairCount[mkPair(256, 'a')]; got != 1 {
		t.Errorf("count(256,'a') = %d, want 1", got)
	}
	if got := tr.pairCount[mkPair('a', 'a')]; got != 0 {
		t.Errorf("count(a,a) = %d, want 0", got)
	}
	if !reflect.DeepEqual(merges, trainNaive(toks, freqs, 1)) {
		t.Errorf("incremental disagrees with naive on the overlap case")
	}
}

// Equal counts must break ties by the smallest packed pair, or the vocab is
// not reproducible across runs.
func TestDeterministicTieBreak(t *testing.T) {
	// "ab" and "cd" both occur once, so (a,b) and (c,d) tie at 1.
	toks := []string{"ab", "cd"}
	freqs := []int64{1, 1}
	for i := 0; i < 20; i++ {
		tr := NewTrainer(toks, freqs, 1, 1<<30)
		m := tr.Train(1, nil)[0]
		if m.A != 'a' || m.B != 'b' {
			t.Fatalf("run %d picked %v; ties must resolve to the smallest pair", i, m)
		}
	}
}

func TestTrainStopsWhenExhausted(t *testing.T) {
	toks := []string{"ab"}
	freqs := []int64{1}
	tr := NewTrainer(toks, freqs, 1, 1<<30)
	if got := tr.Train(100, nil); len(got) != 1 {
		t.Fatalf("got %d merges, want 1 (nothing left to merge)", len(got))
	}
}

func TestEmptyCorpus(t *testing.T) {
	toks, freqs := CountTypes(nil, 4)
	tr := NewTrainer(toks, freqs, 4, 2048)
	if got := tr.Train(10, nil); len(got) != 0 {
		t.Fatalf("got %d merges on an empty corpus", len(got))
	}
}
