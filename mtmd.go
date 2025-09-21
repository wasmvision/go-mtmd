package mtmd

import (
	"runtime"
	"unsafe"

	"github.com/jupiterrider/ffi"
)

// enum mtmd_input_chunk_type
type InputChunkType int32

const (
	InputChunkTypeText InputChunkType = iota
	InputChunkTypeImage
	InputChunkTypeAudio
)

//	struct mtmd_input_text {
//	    const char * text;
//	    bool add_special;
//	    bool parse_special;
//	};
type InputTextType struct {
	Text         *byte
	AddSpecial   bool
	ParseSpecial bool
}

// Opaque types (represented as pointers)
type Context uintptr
type Bitmap uintptr
type ImageTokens uintptr
type InputChunk uintptr
type InputChunks uintptr

//	struct mtmd_context_params {
//	    bool use_gpu;
//	    bool print_timings;
//	    int n_threads;
//	    enum ggml_log_level verbosity;
//	    const char * image_marker; // deprecated, use media_marker instead
//	    const char * media_marker;
//	};
type ContextParamsType struct {
	UseGPU       bool
	PrintTimings bool
	Threads      int
	Verbosity    uint32
	ImageMarker  uintptr
	MediaMarker  uintptr
}

var (
	TypeContextParams    = ffi.NewType(&ffi.TypePointer, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeSint32)
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

	// MTMD_API const char * mtmd_default_marker(void);
	// DefaultMarker     func() string
	// defaultMarkerFunc ffi.Fun

	// MTMD_API struct mtmd_context_params mtmd_context_params_default(void);
	ContextParamsDefault     func() ContextParamsType
	contextParamsDefaultFunc ffi.Fun

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

	// MTMD_API mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
	//                                         const struct llama_model * text_model,
	//                                         const struct mtmd_context_params ctx_params);
	InitFromFile func(mmprojFname string, model LlamaModel, ctxParams ContextParamsType) *Context
)

var currentLib ffi.Lib

func LoadLibrary() {
	var filename string
	switch runtime.GOOS {
	case "linux", "freebsd":
		filename = "./lib/libmtmd.so"
	case "windows":
		filename = "./lib/libmtmd.dll"
	case "darwin":
		filename = "./lib/libmtmd.dylib"
	}

	// load the library
	lib, err := ffi.Load(filename)
	if err != nil {
		panic(err)
	}
	currentLib = lib
}

func Init() {
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

	contextParamsDefaultFunc, err = currentLib.Prep("mtmd_context_params_default", &TypeContextParams)
	if err != nil {
		panic(err)
	}

	ContextParamsDefault = func() ContextParamsType {
		var ctx ContextParamsType
		contextParamsDefaultFunc.Call(unsafe.Pointer(&ctx))
		return ctx
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
