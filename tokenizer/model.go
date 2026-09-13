package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
)

// Model is the on-disk form of a trained tokenizer: just the merge list in
// rank order. Ids 0-255 are implicit (the byte-level base vocab) and merge i
// produces id 256+i, so nothing else needs storing.
type Model struct {
	Version int     `json:"version"`
	Vocab   int     `json:"vocab_size"`
	Merges  [][]int `json:"merges"` // [a, b, new] triples, rank order
}

func SaveModel(path string, merges []Merge) error {
	m := Model{Version: 1, Vocab: 256 + len(merges), Merges: make([][]int, len(merges))}
	for i, mg := range merges {
		m.Merges[i] = []int{int(mg.A), int(mg.B), int(mg.New)}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "")
	if err := enc.Encode(&m); err != nil {
		return err
	}
	return w.Flush()
}

func LoadModel(path string) ([]Merge, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m Model
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	merges := make([]Merge, len(m.Merges))
	for i, t := range m.Merges {
		if len(t) != 3 {
			return nil, fmt.Errorf("model: merge %d is malformed", i)
		}
		merges[i] = Merge{uint32(t[0]), uint32(t[1]), uint32(t[2])}
	}
	return merges, nil
}

// SaveTokens writes ids one per line (diffable, streams back in with a
// scanner) or as little-endian uint32 when binary is set.
func SaveTokens(path string, ids []uint32, binaryFmt bool) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)

	if binaryFmt {
		var b [4]byte
		for _, id := range ids {
			binary.LittleEndian.PutUint32(b[:], id)
			if _, err := w.Write(b[:]); err != nil {
				return err
			}
		}
		return w.Flush()
	}

	var line []byte
	for _, id := range ids {
		line = strconv.AppendUint(line[:0], uint64(id), 10)
		line = append(line, '\n')
		if _, err := w.Write(line); err != nil {
			return err
		}
	}
	return w.Flush()
}

func LoadTokens(path string, binaryFmt bool) ([]uint32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if binaryFmt {
		if len(data)%4 != 0 {
			return nil, fmt.Errorf("tokens: %d bytes is not a multiple of 4", len(data))
		}
		ids := make([]uint32, len(data)/4)
		for i := range ids {
			ids[i] = binary.LittleEndian.Uint32(data[i*4:])
		}
		return ids, nil
	}

	ids := make([]uint32, 0, len(data)/3)
	n := uint64(0)
	have := false
	for _, c := range data {
		if c >= '0' && c <= '9' {
			n = n*10 + uint64(c-'0')
			have = true
			continue
		}
		if have {
			ids = append(ids, uint32(n))
			n, have = 0, false
		}
	}
	if have {
		ids = append(ids, uint32(n))
	}
	return ids, nil
}
