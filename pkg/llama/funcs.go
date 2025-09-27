package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
)

var (
	FFITypeContextParams = ffi.NewType(&ffi.TypeUint32, &ffi.TypeUint32, &ffi.TypeUint32, &ffi.TypeUint32,
		&ffi.TypeSint32, &ffi.TypeSint32,
		&ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeSint32,
		&ffi.TypeFloat, &ffi.TypeFloat, &ffi.TypeFloat, &ffi.TypeFloat, &ffi.TypeFloat, &ffi.TypeFloat,
		&ffi.TypeUint32, &ffi.TypeFloat,
		&ffi.TypePointer, &ffi.TypePointer,
		&ffi.TypeSint32, &ffi.TypeSint32,
		&ffi.TypePointer, &ffi.TypePointer,
		&ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8)
)

var (
	backendInitFunc ffi.Fun

	backendFreeFunc ffi.Fun

	// GGML_API void ggml_backend_load_all(void);
	ggmlBackendLoadAllFunc ffi.Fun

	// LLAMA_API struct llama_context_params        llama_context_default_params(void);
	contextDefaultParamsFunc ffi.Fun

	// LLAMA_API void llama_free(struct llama_context * ctx);
	freeFunc ffi.Fun

	// LLAMA_API void llama_set_warmup(struct llama_context * ctx, bool warmup);
	setWarmupFunc ffi.Fun

	// LLAMA_API int32_t llama_encode(
	//         	struct llama_context * ctx,
	//           struct llama_batch   batch);
	encodeFunc ffi.Fun

	// LLAMA_API int32_t llama_decode(
	// 	struct llama_context * ctx,
	// 		struct llama_batch   batch);
	decodeFunc ffi.Fun

	// LLAMA_API void llama_memory_clear(
	// 	llama_memory_t mem,
	// 				bool data);
	memoryClearFunc ffi.Fun

	// LLAMA_API           llama_memory_t   llama_get_memory  (const struct llama_context * ctx);
	getMemoryFunc ffi.Fun

	// LLAMA_API void llama_synchronize(struct llama_context * ctx);
	synchronizeFunc ffi.Fun

	// LLAMA_API void                           llama_perf_context_reset(      struct llama_context * ctx);
	perfContextResetFunc ffi.Fun
)

func loadFuncs(lib ffi.Lib) {
	var err error
	if backendInitFunc, err = lib.Prep("llama_backend_init", &ffi.TypeVoid); err != nil {
		panic(err)
	}

	if backendFreeFunc, err = lib.Prep("llama_backend_init", &ffi.TypeVoid); err != nil {
		panic(err)
	}

	if ggmlBackendLoadAllFunc, err = lib.Prep("ggml_backend_load_all", &ffi.TypeVoid); err != nil {
		panic(err)
	}

	if contextDefaultParamsFunc, err = lib.Prep("llama_context_default_params", &FFITypeContextParams); err != nil {
		panic(err)
	}

	if freeFunc, err = lib.Prep("llama_free", &ffi.TypeVoid, &ffi.TypePointer); err != nil {
		panic(err)
	}

	if setWarmupFunc, err = lib.Prep("llama_set_warmup", &ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeUint8); err != nil {
		panic(err)
	}

	if encodeFunc, err = lib.Prep("llama_encode", &ffi.TypeSint32, &ffi.TypePointer, &FFITypeBatch); err != nil {
		panic(err)
	}

	if decodeFunc, err = lib.Prep("llama_decode", &ffi.TypeSint32, &ffi.TypePointer, &FFITypeBatch); err != nil {
		panic(err)
	}

	if memoryClearFunc, err = lib.Prep("llama_memory_clear", &ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeUint8); err != nil {
		panic(err)
	}

	if getMemoryFunc, err = lib.Prep("llama_get_memory", &ffi.TypePointer, &ffi.TypePointer); err != nil {
		panic(err)
	}

	if synchronizeFunc, err = lib.Prep("llama_synchronize", &ffi.TypeVoid, &ffi.TypePointer); err != nil {
		panic(err)
	}

	if perfContextResetFunc, err = lib.Prep("llama_perf_context_reset", &ffi.TypeVoid, &ffi.TypePointer); err != nil {
		panic(err)
	}
}

func BackendInit() {
	backendInitFunc.Call(nil)
}

func BackendFree() {
	backendFreeFunc.Call(nil)
}

func GGMLBackendLoadAll() {
	ggmlBackendLoadAllFunc.Call(nil)
}

func ContextDefaultParams() ContextParams {
	var p ContextParams
	contextDefaultParamsFunc.Call(unsafe.Pointer(&p))
	return p
}

func Free(ctx Context) {
	freeFunc.Call(nil, unsafe.Pointer(&ctx))
}

func SetWarmup(ctx Context, warmup bool) {
	setWarmupFunc.Call(nil, unsafe.Pointer(&ctx), &warmup)
}

func Encode(ctx Context, batch Batch) int32 {
	var result ffi.Arg
	encodeFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&ctx), unsafe.Pointer(&batch))

	return int32(result)
}

func Decode(ctx Context, batch Batch) int32 {
	var result ffi.Arg
	decodeFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&ctx), unsafe.Pointer(&batch))

	return int32(result)
}

func MemoryClear(mem Memory, data bool) {
	memoryClearFunc.Call(nil, unsafe.Pointer(&mem), unsafe.Pointer(&data))
}

func GetMemory(ctx Context) Memory {
	var mem Memory
	getMemoryFunc.Call(unsafe.Pointer(&mem), unsafe.Pointer(&ctx))

	return mem
}

func Synchronize(ctx Context) {
	synchronizeFunc.Call(nil, unsafe.Pointer(&ctx))
}

func PerfContextReset(ctx Context) {
	perfContextResetFunc.Call(nil, unsafe.Pointer(&ctx))
}
