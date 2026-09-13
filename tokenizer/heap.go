package main

// A slice-based max-heap over (count, packed pair), hand-rolled rather than
// container/heap: the interface dispatch on Less/Swap is measurable in a loop
// this hot. Entries are never removed on update — the trainer pushes a fresh
// entry on every count change and discards stale pops (lazy deletion).
type heapEntry struct {
	count int64
	pair  uint64
}

type pairHeap struct {
	a []heapEntry
}

// higher reports whether i outranks j: larger count first, ties broken by the
// smaller packed pair. The tie-break is what makes a trained vocab
// reproducible across runs and diffable against a reference implementation.
func (h *pairHeap) higher(i, j int) bool {
	if h.a[i].count != h.a[j].count {
		return h.a[i].count > h.a[j].count
	}
	return h.a[i].pair < h.a[j].pair
}

func (h *pairHeap) Len() int { return len(h.a) }

func (h *pairHeap) Push(count int64, pair uint64) {
	h.a = append(h.a, heapEntry{count, pair})
	i := len(h.a) - 1
	for i > 0 {
		p := (i - 1) / 2
		if !h.higher(i, p) {
			break
		}
		h.a[i], h.a[p] = h.a[p], h.a[i]
		i = p
	}
}

func (h *pairHeap) Pop() (heapEntry, bool) {
	if len(h.a) == 0 {
		return heapEntry{}, false
	}
	top := h.a[0]
	last := len(h.a) - 1
	h.a[0] = h.a[last]
	h.a = h.a[:last]
	i, n := 0, last
	for {
		l, r := 2*i+1, 2*i+2
		best := i
		if l < n && h.higher(l, best) {
			best = l
		}
		if r < n && h.higher(r, best) {
			best = r
		}
		if best == i {
			break
		}
		h.a[i], h.a[best] = h.a[best], h.a[i]
		i = best
	}
	return top, true
}
