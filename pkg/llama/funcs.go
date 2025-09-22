package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
)

var (
	TypeModelParams = ffi.NewType(&ffi.TypePointer, &ffi.TypePointer, &ffi.TypeSint32,
		&ffi.TypeSint32, &ffi.TypeSint32,
		&ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer,
		&ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8)
)

var (
	BackendInit     func()
	backendInitFunc ffi.Fun

	BackendFree     func()
	backendFreeFunc ffi.Fun

	// GGML_API void ggml_backend_load_all(void);
	GGMLBackendLoadAll     func()
	ggmlBackendLoadAllFunc ffi.Fun

	// LLAMA_API struct llama_model_params          llama_model_default_params(void);
	ModelDefaultParams     func() ModelParams
	modelDefaultParamsFunc ffi.Fun

	// LLAMA_API struct llama_model * llama_model_load_from_file(
	//                          const char * path_model,
	//           				struct llama_model_params   params);
	ModelLoadFromFile     func(pathModel string, params ModelParams) Model
	modelLoadFromFileFunc ffi.Fun

	// LLAMA_API struct llama_model_params          llama_model_default_params(void);
	ModelFree     func(model Model)
	modelFreeFunc ffi.Fun

	// LLAMA_API const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
	ModelGetVocab     func(model Model) Vocab
	modelGetVocabFunc ffi.Fun

	// LLAMA_API struct llama_batch llama_batch_init(
	//         int32_t n_tokens,
	BatchInit     func(nTokens int32, embd int32, nSeqMax int32) Batch
	batchInitFunc ffi.Fun

	// LLAMA_API void llama_batch_free(struct llama_batch batch);
	BatchFree     func(batch Batch)
	batchFreeFunc ffi.Fun
)

func Init(currentLib ffi.Lib) {
	var err error
	backendInitFunc, err = currentLib.Prep("llama_backend_init", &ffi.TypeVoid)
	if err != nil {
		panic(err)
	}

	BackendInit = func() {
		backendInitFunc.Call(nil)
	}

	backendFreeFunc, err = currentLib.Prep("llama_backend_init", &ffi.TypeVoid)
	if err != nil {
		panic(err)
	}

	BackendFree = func() {
		backendFreeFunc.Call(nil)
	}

	ggmlBackendLoadAllFunc, err = currentLib.Prep("ggml_backend_load_all", &ffi.TypeVoid)
	if err != nil {
		panic(err)
	}

	GGMLBackendLoadAll = func() {
		ggmlBackendLoadAllFunc.Call(nil)
	}

	modelDefaultParamsFunc, err = currentLib.Prep("llama_model_default_params", &TypeModelParams)
	if err != nil {
		panic(err)
	}

	ModelDefaultParams = func() ModelParams {
		var p ModelParams
		modelDefaultParamsFunc.Call(unsafe.Pointer(&p))
		return p
	}

	modelLoadFromFileFunc, err = currentLib.Prep("llama_model_load_from_file", &ffi.TypePointer, &ffi.TypePointer, &TypeModelParams)
	if err != nil {
		panic(err)
	}

	ModelLoadFromFile = func(pathModel string, params ModelParams) Model {
		var model Model
		file := &[]byte(pathModel + "\x00")[0]
		modelLoadFromFileFunc.Call(unsafe.Pointer(&model), unsafe.Pointer(&file), unsafe.Pointer(&params))
		return model
	}

	modelFreeFunc, err = currentLib.Prep("llama_model_free", &ffi.TypeVoid, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	ModelFree = func(model Model) {
		modelFreeFunc.Call(nil, unsafe.Pointer(&model))
	}

	batchInitFunc, err = currentLib.Prep("llama_batch_init", &ffi.TypePointer, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeSint32)
	if err != nil {
		panic(err)
	}

	BatchInit = func(nTokens int32, embd int32, nSeqMax int32) Batch {
		var batch Batch
		batchInitFunc.Call(unsafe.Pointer(&batch), nTokens, embd, nSeqMax)

		return batch
	}

	batchFreeFunc, err = currentLib.Prep("llama_batch_free", &ffi.TypeVoid, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	BatchFree = func(batch Batch) {
		batchFreeFunc.Call(nil, unsafe.Pointer(&batch))
	}
}
