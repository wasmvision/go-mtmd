package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
)

var (
	TypeLlamaModelParams = ffi.NewType(&ffi.TypePointer, &ffi.TypePointer, &ffi.TypeSint32,
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
	LlamaModelDefaultParams     func() LlamaModelParams
	llamaModelDefaultParamsFunc ffi.Fun

	// LLAMA_API struct llama_model * llama_model_load_from_file(
	//                          const char * path_model,
	//           				struct llama_model_params   params);
	LlamaModelLoadFromFile     func(pathModel string, params LlamaModelParams) LlamaModel
	llamaModelLoadFromFileFunc ffi.Fun

	// LLAMA_API struct llama_model_params          llama_model_default_params(void);
	LlamaModelFree     func(model LlamaModel)
	llamaModelFreeFunc ffi.Fun
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

	llamaModelDefaultParamsFunc, err = currentLib.Prep("llama_model_default_params", &TypeLlamaModelParams)
	if err != nil {
		panic(err)
	}

	LlamaModelDefaultParams = func() LlamaModelParams {
		var p LlamaModelParams
		llamaModelDefaultParamsFunc.Call(unsafe.Pointer(&p))
		return p
	}

	llamaModelLoadFromFileFunc, err = currentLib.Prep("llama_model_load_from_file", &ffi.TypePointer, &ffi.TypePointer, &TypeLlamaModelParams)
	if err != nil {
		panic(err)
	}

	LlamaModelLoadFromFile = func(pathModel string, params LlamaModelParams) LlamaModel {
		var model LlamaModel
		file := &[]byte(pathModel + "\x00")[0]
		llamaModelLoadFromFileFunc.Call(unsafe.Pointer(&model), unsafe.Pointer(&file), unsafe.Pointer(&params))
		return model
	}

	llamaModelFreeFunc, err = currentLib.Prep("llama_model_free", &ffi.TypeVoid, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	LlamaModelFree = func(model LlamaModel) {
		llamaModelFreeFunc.Call(nil, unsafe.Pointer(&model))
	}

	// defaultMarkerFunc, err = currentLib.Prep("mtmd_default_marker", &ffi.TypeUint8)
	// if err != nil {
	// 	panic(err)
	// }

	// DefaultMarker = func() string {
	// 	var result ffi.Arg
	// 	defaultMarkerFunc.Call(result)

	// 	return unix.BytePtrToString(result)
	// }
}
