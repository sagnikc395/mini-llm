//go:build unix

package main

import (
	"os"
	"syscall"
)

// mmapFile maps a file read-only so the corpus never gets copied onto the Go
// heap (and so the GC never has to look at it). Returns the bytes and a close
// func.
func mmapFile(path string) ([]byte, func() error, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	size := st.Size()
	if size == 0 {
		return nil, func() error { return nil }, nil
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, err
	}
	return data, func() error { return syscall.Munmap(data) }, nil
}
