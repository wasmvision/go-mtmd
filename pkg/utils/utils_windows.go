package utils

import "golang.org/x/sys/windows"

func BytePtrFromString(s string) (*byte, error) {
	return windows.BytePtrFromString(s)
}

func BytePtrToString(p *byte) string {
	return windows.BytePtrToString(p)
}
