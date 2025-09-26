package mtmd

import "github.com/jupiterrider/ffi"

func Load(lib ffi.Lib) {
	loadFuncs(lib)
	loadBitmapFuncs(lib)
}
