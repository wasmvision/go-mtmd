package loader

import (
	"path/filepath"
	"runtime"

	"github.com/jupiterrider/ffi"
)

func LoadLibrary(path string) (ffi.Lib, error) {
	var filename string
	switch runtime.GOOS {
	case "linux", "freebsd":
		filename = filepath.Join(path, "libmtmd.so")
	case "windows":
		filename = filepath.Join(path, "libmtmd.dll")
	case "darwin":
		filename = filepath.Join(path, "libmtmd.dylib")
	}

	return ffi.Load(filename)
}
