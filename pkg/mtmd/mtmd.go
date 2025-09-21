package mtmd

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
	"github.com/wasmvision/go-mtmd/pkg/llama"
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
	TypeContextParams = ffi.NewType(&ffi.TypePointer, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeSint32)
)

var (
	// MTMD_API const char * mtmd_default_marker(void);
	// DefaultMarker     func() string
	// defaultMarkerFunc ffi.Fun

	// MTMD_API struct mtmd_context_params mtmd_context_params_default(void);
	ContextParamsDefault     func() ContextParamsType
	contextParamsDefaultFunc ffi.Fun

	// MTMD_API mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
	//                                         const struct llama_model * text_model,
	//                                         const struct mtmd_context_params ctx_params);
	InitFromFile func(mmprojFname string, model llama.LlamaModel, ctxParams ContextParamsType) *Context
)

func Init(currentLib ffi.Lib) {
	var err error

	contextParamsDefaultFunc, err = currentLib.Prep("mtmd_context_params_default", &TypeContextParams)
	if err != nil {
		panic(err)
	}

	ContextParamsDefault = func() ContextParamsType {
		var ctx ContextParamsType
		contextParamsDefaultFunc.Call(unsafe.Pointer(&ctx))
		return ctx
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
