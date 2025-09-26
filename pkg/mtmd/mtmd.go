package mtmd

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
	"github.com/wasmvision/yzma/pkg/llama"
	"golang.org/x/sys/unix"
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
type InputText struct {
	Text         *byte
	AddSpecial   bool
	ParseSpecial bool
}

// Opaque types (represented as pointers)
type Context uintptr
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
	Threads      int32
	Verbosity    llama.LogLevel
	ImageMarker  *byte
	MediaMarker  *byte
}

var (
	FFITypeContextParams = ffi.NewType(&ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	TypeInputText        = ffi.NewType(&ffi.TypePointer, &ffi.TypeUint8, &ffi.TypeUint8)
)

var (
	// MTMD_API const char * mtmd_default_marker(void);
	DefaultMarker     func() string
	defaultMarkerFunc ffi.Fun

	// MTMD_API struct mtmd_context_params mtmd_context_params_default(void);
	ContextParamsDefault     func() ContextParamsType
	contextParamsDefaultFunc ffi.Fun

	// MTMD_API mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
	//                                         const struct llama_model * text_model,
	//                                         const struct mtmd_context_params ctx_params);
	InitFromFile     func(mmprojFname string, model llama.Model, ctxParams ContextParamsType) Context
	initFromFileFunc ffi.Fun

	// MTMD_API void mtmd_free(mtmd_context * ctx);
	Free     func(ctx Context)
	freeFunc ffi.Fun

	// MTMD_API bool mtmd_support_vision(mtmd_context * ctx);
	SupportVision     func(ctx Context) bool
	supportVisionFunc ffi.Fun

	// MTMD_API mtmd_input_chunks *      mtmd_input_chunks_init(void);
	InputChunksInit     func() InputChunks
	inputChunksInitFunc ffi.Fun

	// MTMD_API size_t mtmd_input_chunks_size(const mtmd_input_chunks * chunks);
	InputChunksSize     func(chunks InputChunks) uint32
	inputChunksSizeFunc ffi.Fun

	// MTMD_API int32_t mtmd_tokenize(mtmd_context * ctx,
	//                            mtmd_input_chunks * output,
	//                            const mtmd_input_text * text,
	//                            const mtmd_bitmap ** bitmaps,
	//                            size_t n_bitmaps);
	Tokenize     func(ctx Context, out InputChunks, text *InputText, bitmaps *Bitmap, nBitmaps uint64) int32
	tokenizeFunc ffi.Fun

	// MTMD_API int32_t mtmd_helper_eval_chunks(mtmd_context * ctx,
	//                                          struct llama_context * lctx,
	//                                          const mtmd_input_chunks * chunks,
	//                                          llama_pos n_past,
	//                                          llama_seq_id seq_id,
	//                                          int32_t n_batch,
	//                                          bool logits_last,
	//                                          llama_pos * new_n_past);
	HelperEvalChunks     func(ctx Context, lctx llama.Context, chunks InputChunks, nPast llama.Pos, seqID llama.SeqId, nBatch int32, logitsLast bool, newNPast *llama.Pos) int32
	helperEvalChunksFunc ffi.Fun
)

func loadFuncs(currentLib ffi.Lib) {
	var err error

	defaultMarkerFunc, err = currentLib.Prep("mtmd_default_marker", &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	DefaultMarker = func() string {
		var marker *byte
		defaultMarkerFunc.Call(unsafe.Pointer(&marker))
		return unix.BytePtrToString(marker)
	}

	contextParamsDefaultFunc, err = currentLib.Prep("mtmd_context_params_default", &FFITypeContextParams)
	if err != nil {
		panic(err)
	}

	ContextParamsDefault = func() ContextParamsType {
		var ctx ContextParamsType
		contextParamsDefaultFunc.Call(unsafe.Pointer(&ctx))
		return ctx
	}

	initFromFileFunc, err = currentLib.Prep("mtmd_init_from_file", &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer, &FFITypeContextParams)
	if err != nil {
		panic(err)
	}

	InitFromFile = func(mmprojFname string, model llama.Model, ctxParams ContextParamsType) Context {
		var ctx Context
		file := &[]byte(mmprojFname + "\x00")[0]
		initFromFileFunc.Call(unsafe.Pointer(&ctx), unsafe.Pointer(&file), unsafe.Pointer(&model), unsafe.Pointer(&ctxParams))
		return ctx
	}

	freeFunc, err = currentLib.Prep("mtmd_free", &ffi.TypeVoid, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	Free = func(ctx Context) {
		freeFunc.Call(nil, unsafe.Pointer(&ctx))
	}

	supportVisionFunc, err = currentLib.Prep("mtmd_support_vision", &ffi.TypeUint8, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	SupportVision = func(ctx Context) bool {
		var result ffi.Arg
		supportVisionFunc.Call(&result, unsafe.Pointer(&ctx))

		return result.Bool()
	}

	inputChunksInitFunc, err = currentLib.Prep("mtmd_input_chunks_init", &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	InputChunksInit = func() InputChunks {
		var chunks InputChunks
		inputChunksInitFunc.Call(unsafe.Pointer(&chunks))

		return chunks
	}

	inputChunksSizeFunc, err = currentLib.Prep("mtmd_input_chunks_size", &ffi.TypeSint32, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	InputChunksSize = func(chunks InputChunks) uint32 {
		var result ffi.Arg
		inputChunksSizeFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&chunks))

		return uint32(result)
	}

	tokenizeFunc, err = currentLib.Prep("mtmd_tokenize", &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypeUint64)
	if err != nil {
		panic(err)
	}

	Tokenize = func(ctx Context, out InputChunks, text *InputText, bitmaps *Bitmap, nBitmaps uint64) int32 {
		var result ffi.Arg
		tokenizeFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&ctx), unsafe.Pointer(&out), unsafe.Pointer(&text), unsafe.Pointer(&bitmaps), unsafe.Pointer(&nBitmaps))

		return int32(result)
	}

	helperEvalChunksFunc, err = currentLib.Prep("mtmd_helper_eval_chunks", &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer,
		&ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeUint8, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	HelperEvalChunks = func(ctx Context, lctx llama.Context, chunks InputChunks, nPast llama.Pos, seqID llama.SeqId, nBatch int32, logitsLast bool, newNPast *llama.Pos) int32 {
		var result ffi.Arg
		helperEvalChunksFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&ctx), unsafe.Pointer(&lctx), unsafe.Pointer(&chunks), unsafe.Pointer(&nPast), unsafe.Pointer(&seqID),
			unsafe.Pointer(&nBatch), unsafe.Pointer(&logitsLast), unsafe.Pointer(&newNPast))

		return int32(result)
	}
}
