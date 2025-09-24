package llama

import "github.com/jupiterrider/ffi"

func Init(lib ffi.Lib) {
	initFuncs(lib)
	initModel(lib)
	initBatch(lib)
	initVocab(lib)
	initSampling(lib)
}
