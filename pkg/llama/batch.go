package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
)

var (
	FFITypeBatch = ffi.NewType(&ffi.TypeSint32,
		&ffi.TypePointer, &ffi.TypePointer,
		&ffi.TypePointer, &ffi.TypePointer,
		&ffi.TypePointer, &ffi.TypePointer)
)

var (
	// LLAMA_API struct llama_batch llama_batch_init(
	//         int32_t n_tokens,
	batchInitFunc ffi.Fun

	// LLAMA_API void llama_batch_free(struct llama_batch batch);
	batchFreeFunc ffi.Fun

	// LLAMA_API struct llama_batch llama_batch_get_one(
	//               llama_token * tokens,
	//                   int32_t   n_tokens);
	batchGetOneFunc ffi.Fun
)

func loadBatchFuncs(lib ffi.Lib) {
	var err error

	if batchInitFunc, err = lib.Prep("llama_batch_init", &FFITypeBatch, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeSint32); err != nil {
		panic(err)
	}

	if batchFreeFunc, err = lib.Prep("llama_batch_free", &ffi.TypeVoid, &ffi.TypePointer); err != nil {
		panic(err)
	}

	if batchGetOneFunc, err = lib.Prep("llama_batch_get_one", &FFITypeBatch, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		panic(err)
	}
}

func BatchInit(nTokens int32, embd int32, nSeqMax int32) Batch {
	var batch Batch
	batchInitFunc.Call(unsafe.Pointer(&batch), unsafe.Pointer(&nTokens), unsafe.Pointer(&embd), unsafe.Pointer(&nSeqMax))

	return batch
}

func BatchFree(batch Batch) {
	batchFreeFunc.Call(nil, unsafe.Pointer(&batch))
}

func BatchGetOne(tokens []Token) Batch {
	toks := unsafe.SliceData(tokens)
	nTokens := int32(len(tokens))

	var batch Batch
	batchGetOneFunc.Call(unsafe.Pointer(&batch), unsafe.Pointer(&toks), &nTokens)

	return batch
}
