package mtmd

import "github.com/jupiterrider/ffi"

func Load(lib ffi.Lib) error {
	loadFuncs(lib)
	loadBitmapFuncs(lib)

	return nil
}
