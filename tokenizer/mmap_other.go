//go:build !unix

package main

import "os"

// mmapFile falls back to a plain read where mmap is unavailable.
func mmapFile(path string) ([]byte, func() error, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	return data, func() error { return nil }, nil
}
