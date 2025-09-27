package llama

import "github.com/jupiterrider/ffi"

func Load(lib ffi.Lib) {
	loadFuncs(lib)
	loadModelFuncs(lib)
	loadBatchFuncs(lib)
	loadVocabFuncs(lib)
	loadSamplingFuncs(lib)
	loadChatFuncs(lib)
}
