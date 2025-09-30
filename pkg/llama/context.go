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

	// LLAMA_API void                           llama_perf_context_reset(      struct llama_context * ctx);
	perfContextResetFunc ffi.Fun
)

func loadContextFuncs(lib ffi.Lib) error {
	var err error
	if contextDefaultParamsFunc, err = lib.Prep("llama_context_default_params", &FFITypeContextParams); err != nil {
		return err
	}

	if freeFunc, err = lib.Prep("llama_free", &ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return err
	}

	if setWarmupFunc, err = lib.Prep("llama_set_warmup", &ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeUint8); err != nil {
		return err
	}

	if encodeFunc, err = lib.Prep("llama_encode", &ffi.TypeSint32, &ffi.TypePointer, &FFITypeBatch); err != nil {
		return err
	}

	if decodeFunc, err = lib.Prep("llama_decode", &ffi.TypeSint32, &ffi.TypePointer, &FFITypeBatch); err != nil {
		return err
	}

	if perfContextResetFunc, err = lib.Prep("llama_perf_context_reset", &ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return err
	}

	return nil
}

// ContextDefaultParams returns the default params to initialize a model context.
func ContextDefaultParams() ContextParams {
	var p ContextParams
	contextDefaultParamsFunc.Call(unsafe.Pointer(&p))
	return p
}

// Free frees the resources for a model context.
func Free(ctx Context) {
	freeFunc.Call(nil, unsafe.Pointer(&ctx))
}

// SetWarmup sets the model context warmup mode on or off.
func SetWarmup(ctx Context, warmup bool) {
	setWarmupFunc.Call(nil, unsafe.Pointer(&ctx), &warmup)
}

// Encode encodes a batch of Token.
func Encode(ctx Context, batch Batch) int32 {
	var result ffi.Arg
	encodeFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&ctx), unsafe.Pointer(&batch))

	return int32(result)
}

// Decode decodes a batch of Token.
func Decode(ctx Context, batch Batch) int32 {
	var result ffi.Arg
	decodeFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&ctx), unsafe.Pointer(&batch))

	return int32(result)
}

// PerfContextReset resets the performance metrics for the model context.
func PerfContextReset(ctx Context) {
	perfContextResetFunc.Call(nil, unsafe.Pointer(&ctx))
}
