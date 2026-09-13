package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/debug"
	"time"
)

const usage = `mini-llm tokenizer — multi-threaded byte-level BPE

usage:
  tokenizer train  -in corpus.txt -out model.json [-vocab 4096] [-threads N]
  tokenizer encode -model model.json -in text.txt -out tokens.txt [-binary]
  tokenizer decode -model model.json -in tokens.txt -out text.txt [-binary]
  tokenizer bench  -in corpus.txt [-vocab 4096] [-threads N]

run any subcommand with -h for its flags.
`

func main() {
	log.SetFlags(0)
	if len(os.Args) < 2 {
		fmt.Print(usage)
		return
	}
	switch os.Args[1] {
	case "train":
		cmdTrain(os.Args[2:])
	case "encode":
		cmdEncode(os.Args[2:])
	case "decode":
		cmdDecode(os.Args[2:])
	case "bench":
		cmdBench(os.Args[2:])
	case "-h", "--help", "help":
		fmt.Print(usage)
	default:
		fmt.Print(usage)
		os.Exit(2)
	}
}

func mb(n int) float64 { return float64(n) / (1 << 20) }

// Throughput is reported in MB of input text per second, never tokens per
// second — tokens/s conflates speed with compression ratio.
func rate(n int, d time.Duration) float64 {
	if d <= 0 {
		return 0
	}
	return mb(n) / d.Seconds()
}

func cmdTrain(args []string) {
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	in := fs.String("in", "", "corpus file (required)")
	out := fs.String("out", "model.json", "output model path")
	vocab := fs.Int("vocab", 4096, "target vocab size (>= 256)")
	threads := fs.Int("threads", runtime.NumCPU(), "worker goroutines")
	threshold := fs.Int("threshold", 2048, "affected-word count above which a merge iteration goes parallel")
	verbose := fs.Bool("v", false, "print every merge")
	gogc := fs.Int("gogc", 400, "GOGC during training; raising it is usually a clear win here")
	fs.Parse(args)

	if *in == "" {
		fs.Usage()
		os.Exit(2)
	}
	if *vocab <= 256 {
		log.Fatalf("vocab must be > 256 (256 byte values are the base vocab)")
	}
	debug.SetGCPercent(*gogc)

	data, closeFn, err := mmapFile(*in)
	if err != nil {
		log.Fatal(err)
	}
	defer closeFn()

	total := time.Now()

	t0 := time.Now()
	toks, freqs := CountTypes(data, *threads)
	d1 := time.Since(t0)
	nTok := int64(0)
	for _, f := range freqs {
		nTok += f
	}
	fmt.Printf("phase 1  %7.2f MB -> %d types (%d pre-tokens)  %v  %.1f MB/s\n",
		mb(len(data)), len(toks), nTok, d1.Round(time.Millisecond), rate(len(data), d1))

	t0 = time.Now()
	tr := NewTrainer(toks, freqs, *threads, *threshold)
	d2 := time.Since(t0)
	fmt.Printf("phase 2a arena + initial pair count: %d distinct pairs  %v\n", len(tr.pairCount), d2.Round(time.Millisecond))

	t0 = time.Now()
	var progress func(int, Merge, int64)
	if *verbose {
		progress = func(rank int, m Merge, c int64) {
			fmt.Printf("  merge %5d: (%d,%d) -> %d  count %d\n", rank, m.A, m.B, m.New, c)
		}
	}
	merges := tr.Train(*vocab-256, progress)
	d3 := time.Since(t0)
	fmt.Printf("phase 2b %d merges  %v  (%.0f merges/s)\n",
		len(merges), d3.Round(time.Millisecond), float64(len(merges))/d3.Seconds())

	if err := SaveModel(*out, merges); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("wrote %s (vocab %d) — total %v, %.1f MB/s end to end\n",
		*out, 256+len(merges), time.Since(total).Round(time.Millisecond), rate(len(data), time.Since(total)))
}

func cmdEncode(args []string) {
	fs := flag.NewFlagSet("encode", flag.ExitOnError)
	model := fs.String("model", "model.json", "model file")
	in := fs.String("in", "", "input text (required)")
	out := fs.String("out", "tokens.txt", "output token file")
	binaryFmt := fs.Bool("binary", false, "write little-endian uint32 instead of one id per line")
	threads := fs.Int("threads", runtime.NumCPU(), "worker goroutines")
	fs.Parse(args)
	if *in == "" {
		fs.Usage()
		os.Exit(2)
	}

	merges, err := LoadModel(*model)
	if err != nil {
		log.Fatal(err)
	}
	enc := NewEncoder(merges)

	data, closeFn, err := mmapFile(*in)
	if err != nil {
		log.Fatal(err)
	}
	defer closeFn()

	t0 := time.Now()
	ids := enc.Encode(data, *threads)
	d := time.Since(t0)

	if err := SaveTokens(*out, ids, *binaryFmt); err != nil {
		log.Fatal(err)
	}
	ratio := 0.0
	if len(ids) > 0 {
		ratio = float64(len(data)) / float64(len(ids))
	}
	fmt.Printf("encoded %.2f MB -> %d tokens in %v (%.1f MB/s), %.2f bytes/token, wrote %s\n",
		mb(len(data)), len(ids), d.Round(time.Millisecond), rate(len(data), d), ratio, *out)
}

func cmdDecode(args []string) {
	fs := flag.NewFlagSet("decode", flag.ExitOnError)
	model := fs.String("model", "model.json", "model file")
	in := fs.String("in", "", "token file (required)")
	out := fs.String("out", "decoded.txt", "output text file")
	binaryFmt := fs.Bool("binary", false, "input is little-endian uint32")
	fs.Parse(args)
	if *in == "" {
		fs.Usage()
		os.Exit(2)
	}

	merges, err := LoadModel(*model)
	if err != nil {
		log.Fatal(err)
	}
	ids, err := LoadTokens(*in, *binaryFmt)
	if err != nil {
		log.Fatal(err)
	}
	text := NewEncoder(merges).Decode(ids)
	if err := os.WriteFile(*out, text, 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("decoded %d tokens -> %.2f MB, wrote %s\n", len(ids), mb(len(text)), *out)
}

// cmdBench trains and then encodes the same corpus, reporting per-phase
// throughput at 1..N threads so the scaling is visible.
func cmdBench(args []string) {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	in := fs.String("in", "", "corpus file (required)")
	vocab := fs.Int("vocab", 4096, "target vocab size")
	maxThreads := fs.Int("threads", runtime.NumCPU(), "highest thread count to try")
	threshold := fs.Int("threshold", 2048, "parallel gate for the merge loop")
	fs.Parse(args)
	if *in == "" {
		fs.Usage()
		os.Exit(2)
	}
	debug.SetGCPercent(400)

	data, closeFn, err := mmapFile(*in)
	if err != nil {
		log.Fatal(err)
	}
	defer closeFn()
	fmt.Printf("corpus %.2f MB, vocab %d\n\n", mb(len(data)), *vocab)
	fmt.Printf("%8s %14s %14s %14s %14s\n", "threads", "count MB/s", "merge s", "encode MB/s", "bytes/token")

	for n := 1; n <= *maxThreads; n *= 2 {
		t0 := time.Now()
		toks, freqs := CountTypes(data, n)
		dCount := time.Since(t0)

		t0 = time.Now()
		tr := NewTrainer(toks, freqs, n, *threshold)
		merges := tr.Train(*vocab-256, nil)
		dTrain := time.Since(t0)

		enc := NewEncoder(merges)
		t0 = time.Now()
		ids := enc.Encode(data, n)
		dEnc := time.Since(t0)

		ratio := 0.0
		if len(ids) > 0 {
			ratio = float64(len(data)) / float64(len(ids))
		}
		fmt.Printf("%8d %14.1f %14.2f %14.1f %14.2f\n",
			n, rate(len(data), dCount), dTrain.Seconds(), rate(len(data), dEnc), ratio)
	}
}
