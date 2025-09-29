package mtmd

import (
	"sync"

	"github.com/jupiterrider/ffi"
)

var muHelperEvalChunks sync.Mutex

func Load(lib ffi.Lib) error {
	loadFuncs(lib)
	loadBitmapFuncs(lib)

	return nil
}
