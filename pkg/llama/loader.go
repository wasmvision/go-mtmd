package llama

import "github.com/jupiterrider/ffi"

func Init(lib ffi.Lib) {
	initFuncs(lib)
	initSampling(lib)
}
