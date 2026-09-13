package main

import (
	"math/rand"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// benchCorpus builds text with a realistic type/token profile: a large
// vocabulary sampled Zipfian, so phase 1 sees millions of pre-tokens over tens
// of thousands of types and phase 2 has a real merge loop to run. The tiny
// fixed pool used by the correctness tests would make both phases trivial.
func benchCorpus(n int) []byte {
	r := rand.New(rand.NewSource(1))
	syll := []string{"ka", "ro", "mi", "ta", "len", "dor", "fi", "sha", "ven", "tri",
		"nos", "bel", "qua", "zim", "pol", "and", "ing", "tion", "est", "ly"}
	const V = 50_000
	vocab := make([]string, V)
	for i := range vocab {
		w := make([]byte, 0, 12)
		for j := r.Intn(4) + 1; j > 0; j-- {
			w = append(w, syll[r.Intn(len(syll))]...)
		}
		vocab[i] = string(w)
	}
	// Inverse-CDF sampling of a 1/rank distribution.
	cdf := make([]float64, V)
	sum := 0.0
	for i := range cdf {
		sum += 1 / float64(i+1)
		cdf[i] = sum
	}
	var b strings.Builder
	b.Grow(n + 64)
	for words := 0; b.Len() < n; words++ {
		i := sort.SearchFloat64s(cdf, r.Float64()*sum)
		if i >= V {
			i = V - 1
		}
		b.WriteString(vocab[i])
		if words%13 == 12 {
			b.WriteByte('\n')
		} else {
			b.WriteByte(' ')
		}
	}
	return []byte(b.String())
}

// Throughput is reported in MB/s of input text (b.SetBytes), never tokens/s —
// tokens/s conflates speed with compression ratio.
func BenchmarkPretokenize(b *testing.B) {
	data := benchCorpus(8 << 20)
	b.SetBytes(int64(len(data)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := scanner{b: data}
		for {
			if _, _, ok := s.next(); !ok {
				break
			}
		}
	}
}

func BenchmarkCountTypes(b *testing.B) {
	data := benchCorpus(8 << 20)
	for _, w := range threadCounts() {
		b.Run("threads="+strconv.Itoa(w), func(b *testing.B) {
			b.SetBytes(int64(len(data)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				CountTypes(data, w)
			}
		})
	}
}

func BenchmarkTrain(b *testing.B) {
	data := benchCorpus(8 << 20)
	toks, freqs := CountTypes(data, runtime.NumCPU())
	for _, w := range threadCounts() {
		b.Run("threads="+strconv.Itoa(w), func(b *testing.B) {
			b.SetBytes(int64(len(data)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				NewTrainer(toks, freqs, w, 2048).Train(1024, nil)
			}
		})
	}
}

// Sweep the parallel gate for the merge loop; the affected set collapses fast,
// so past a few hundred merges the delta reduction costs more than it saves.
func BenchmarkTrainThreshold(b *testing.B) {
	data := benchCorpus(8 << 20)
	toks, freqs := CountTypes(data, runtime.NumCPU())
	for _, th := range []int{0, 256, 1024, 2048, 8192, 1 << 30} {
		b.Run("threshold="+strconv.Itoa(th), func(b *testing.B) {
			b.SetBytes(int64(len(data)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				NewTrainer(toks, freqs, runtime.NumCPU(), th).Train(1024, nil)
			}
		})
	}
}

func BenchmarkEncode(b *testing.B) {
	data := benchCorpus(8 << 20)
	toks, freqs := CountTypes(data, runtime.NumCPU())
	enc := NewEncoder(NewTrainer(toks, freqs, runtime.NumCPU(), 2048).Train(4096-256, nil))
	for _, w := range threadCounts() {
		b.Run("threads="+strconv.Itoa(w), func(b *testing.B) {
			b.SetBytes(int64(len(data)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				enc.Encode(data, w)
			}
		})
	}
}

func threadCounts() []int {
	out := []int{}
	for w := 1; w <= runtime.NumCPU(); w *= 2 {
		out = append(out, w)
	}
	if last := out[len(out)-1]; last != runtime.NumCPU() {
		out = append(out, runtime.NumCPU())
	}
	return out
}
