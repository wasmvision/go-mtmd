package mtmd

import "github.com/jupiterrider/ffi"

func Init(lib ffi.Lib) {
	initFuncs(lib)
	initBitmapFuncs(lib)
}
