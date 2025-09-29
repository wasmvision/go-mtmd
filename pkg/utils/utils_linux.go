package utils

import "golang.org/x/sys/unix"

func BytePtrFromString(s string) (*byte, error) {
	return unix.BytePtrFromString(s)
}

func BytePtrToString(p *byte) string {
	return unix.BytePtrToString(p)
}
