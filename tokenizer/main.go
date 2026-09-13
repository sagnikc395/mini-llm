package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
)

// use a struct to represent the most common pairs
type Pair struct {
	A, B int
}

func getStats(tokens []int) map[Pair]int {
	counts := make(map[Pair]int)
	for i := 0; i < len(tokens)-1; i++ {
		counts[Pair{tokens[i], tokens[i+1]}] += 1
	}
	return counts
}

// method for the most common pair
func findMostCommonPair(tokens []int) Pair {
	counts := getStats(tokens)
	mostCommon := Pair{}
	maxCount := -1

	for k, v := range counts {
		if v > maxCount {
			maxCount = v
			mostCommon = k
		}
	}

	return mostCommon
}

// step3: merging tokens ; once we have found the most common pairs, we
// can replace it with a new token

func merge(ids []int, pair Pair, idx int) []int {
	newIds := make([]int, 0)
	i := 0
	for i < len(ids) {
		if i < len(ids)-1 && ids[i] == pair.A && ids[i+1] == pair.B {
			newIds = append(newIds, idx)
			i += 2
		} else {
			newIds = append(newIds, ids[i])
			i += 1
		}
	}
	return newIds
}

// step1: ingestion — raw text -> initial token ids.
// BPE starts from the raw UTF-8 bytes, so every byte of the text
// is an initial token id in [0, 255]. No vocab needed yet.
func ingest(text string) []int {
	raw := []byte(text) // UTF-8 bytes, NOT runes (see note below)
	ids := make([]int, len(raw))
	for i, b := range raw {
		ids[i] = int(b)
	}
	return ids
}

// write the final token ids to a file, one per line. Plain text so the
// output is diffable and can be streamed back in later with a scanner.
func saveTokens(ids []int, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	for _, id := range ids {
		if _, err := w.WriteString(strconv.Itoa(id) + "\n"); err != nil {
			return err
		}
	}
	return w.Flush()
}

func main() {
	// ingestion: either pass a corpus file as the first argument
	// (go run main.go corpus.txt) or fall back to an inline string.
	var text string
	if len(os.Args) > 1 {
		data, err := os.ReadFile(os.Args[1])
		if err != nil {
			log.Fatal(err)
		}
		text = string(data)
	} else {
		text = "the quick brown fox jumps over the lazy dog"
	}

	ids := ingest(text)
	fmt.Printf("ingested %d bytes -> %d initial tokens\n", len(text), len(ids))

	// step4: training loop — repeatedly merge the most common pair
	// with a fresh token id (ids 0-255 are already taken by the bytes).
	const numMerges = 20
	merges := make(map[Pair]int)
	for i := 0; i < numMerges && len(ids) >= 2; i++ {
		pair := findMostCommonPair(ids)
		newId := 256 + i
		ids = merge(ids, pair, newId)
		merges[pair] = newId
		fmt.Printf("merge %2d: %v -> %d (len now %d)\n", i+1, pair, newId, len(ids))
	}

	fmt.Println("final ids:", ids)
	fmt.Printf("compression: %d bytes -> %d tokens\n", len([]byte(text)), len(ids))

	// persist the tokens: pass an output path as the second argument
	// (go run main.go corpus.txt out.txt) or default to tokens.txt.
	outPath := "tokens.txt"
	if len(os.Args) > 2 {
		outPath = os.Args[2]
	}
	if err := saveTokens(ids, outPath); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("wrote %d tokens to %s\n", len(ids), outPath)
}
