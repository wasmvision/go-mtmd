package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
	"golang.org/x/sys/unix"
)

var (
	TypeModelParams = ffi.NewType(&ffi.TypePointer, &ffi.TypePointer, &ffi.TypeSint32,
		&ffi.TypeSint32, &ffi.TypeSint32,
		&ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer,
		&ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8)

	TypeContextParams = ffi.NewType(&ffi.TypeUint32, &ffi.TypeUint32, &ffi.TypeUint32, &ffi.TypeUint32,
		&ffi.TypeSint32, &ffi.TypeSint32,
		&ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeSint32,
		&ffi.TypeFloat, &ffi.TypeFloat, &ffi.TypeFloat, &ffi.TypeFloat, &ffi.TypeFloat, &ffi.TypeFloat,
		&ffi.TypeUint32, &ffi.TypeFloat,
		&ffi.TypePointer, &ffi.TypePointer,
		&ffi.TypeSint32, &ffi.TypeSint32,
		&ffi.TypePointer, &ffi.TypePointer,
		&ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8)

	TypeBatch = ffi.NewType(&ffi.TypeSint32,
		&ffi.TypePointer, &ffi.TypePointer,
		&ffi.TypePointer, &ffi.TypePointer,
		&ffi.TypePointer, &ffi.TypePointer)
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

	// LLAMA_API struct llama_context_params        llama_context_default_params(void);
	ContextDefaultParams     func() ContextParams
	contextDefaultParamsFunc ffi.Fun

	// LLAMA_API struct llama_context * llama_init_from_model(
	//                  struct llama_model * model,
	//         			struct llama_context_params   params);
	InitFromModel     func(model Model, params ContextParams) Context
	initFromModelFunc ffi.Fun

	// LLAMA_API void llama_free(struct llama_context * ctx);
	Free     func(ctx Context)
	freeFunc ffi.Fun

	// LLAMA_API struct llama_batch llama_batch_init(
	//         int32_t n_tokens,
	BatchInit     func(nTokens int32, embd int32, nSeqMax int32) Batch
	batchInitFunc ffi.Fun

	// LLAMA_API void llama_batch_free(struct llama_batch batch);
	BatchFree     func(batch Batch)
	batchFreeFunc ffi.Fun

	// LLAMA_API const char * llama_model_chat_template(const struct llama_model * model, const char * name);
	ModelChatTemplate     func(model Model, name string) string
	modelChatTemplateFunc ffi.Fun

	// LLAMA_API void llama_set_warmup(struct llama_context * ctx, bool warmup);
	SetWarmup     func(ctx Context, warmup bool)
	setWarmupFunc ffi.Fun

	// LLAMA_API llama_token llama_vocab_bos(const struct llama_vocab * vocab); // beginning-of-sentence
	VocabBOS     func(vocab Vocab) Token
	vocabBOSFunc ffi.Fun

	// LLAMA_API llama_token llama_vocab_eos(const struct llama_vocab * vocab); // end-of-sentence
	VocabEOS     func(vocab Vocab) Token
	vocabEOSFunc ffi.Fun

	// LLAMA_API bool llama_model_has_encoder(const struct llama_model * model);
	ModelHasEncoder     func(model Model) bool
	modelHasEncoderFunc ffi.Fun

	// LLAMA_API bool llama_model_has_decoder(const struct llama_model * model);
	ModelHasDecoder     func(model Model) bool
	modelHasDecoderFunc ffi.Fun

	// LLAMA_API int32_t llama_encode(
	//         	struct llama_context * ctx,
	//           struct llama_batch   batch);
	Encode     func(ctx Context, batch Batch) int32
	encodeFunc ffi.Fun

	// LLAMA_API struct llama_batch llama_batch_get_one(
	//               llama_token * tokens,
	//                   int32_t   n_tokens);
	BatchGetOne     func(tokens *Token, nTokens int32) Batch
	batchGetOneFunc ffi.Fun

	// LLAMA_API llama_token llama_model_decoder_start_token(const struct llama_model * model);
	ModelDecoderStartToken     func(model Model) Token
	modelDecoderStartTokenFunc ffi.Fun

	// LLAMA_API int32_t llama_decode(
	// 	struct llama_context * ctx,
	// 		struct llama_batch   batch);
	Decode     func(ctx Context, batch Batch) int32
	decodeFunc ffi.Fun

	// LLAMA_API void llama_memory_clear(
	// 	llama_memory_t mem,
	// 				bool data);
	MemoryClear     func(mem Memory, data bool)
	memoryClearFunc ffi.Fun

	// LLAMA_API           llama_memory_t   llama_get_memory  (const struct llama_context * ctx);
	GetMemory     func(ctx Context) Memory
	getMemoryFunc ffi.Fun

	// LLAMA_API void llama_synchronize(struct llama_context * ctx);
	Synchronize     func(ctx Context)
	synchronizeFunc ffi.Fun

	// LLAMA_API void                           llama_perf_context_reset(      struct llama_context * ctx);
	PerfContextReset     func(ctx Context)
	perfContextResetFunc ffi.Fun
)

func initFuncs(currentLib ffi.Lib) {
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

	modelGetVocabFunc, err = currentLib.Prep("llama_model_get_vocab", &ffi.TypePointer, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	ModelGetVocab = func(model Model) Vocab {
		var vocab Vocab
		modelGetVocabFunc.Call(unsafe.Pointer(&vocab), unsafe.Pointer(&model))

		return vocab
	}

	contextDefaultParamsFunc, err = currentLib.Prep("llama_context_default_params", &TypeContextParams)
	if err != nil {
		panic(err)
	}

	ContextDefaultParams = func() ContextParams {
		var p ContextParams
		contextDefaultParamsFunc.Call(unsafe.Pointer(&p))
		return p
	}

	initFromModelFunc, err = currentLib.Prep("llama_init_from_model", &ffi.TypePointer, &ffi.TypePointer, &TypeContextParams)
	if err != nil {
		panic(err)
	}

	InitFromModel = func(model Model, params ContextParams) Context {
		var ctx Context
		initFromModelFunc.Call(unsafe.Pointer(&ctx), unsafe.Pointer(&model), unsafe.Pointer(&params))

		return ctx
	}

	freeFunc, err = currentLib.Prep("llama_free", &ffi.TypeVoid, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	Free = func(ctx Context) {
		freeFunc.Call(nil, unsafe.Pointer(&ctx))
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

	modelChatTemplateFunc, err = currentLib.Prep("llama_model_chat_template", &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	ModelChatTemplate = func(model Model, name string) string {
		var template *byte
		n := &[]byte(name + "\x00")[0]
		modelChatTemplateFunc.Call(unsafe.Pointer(&template), unsafe.Pointer(&model), unsafe.Pointer(&n))

		return unix.BytePtrToString(template)
	}

	setWarmupFunc, err = currentLib.Prep("llama_set_warmup", &ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeUint8)
	if err != nil {
		panic(err)
	}

	SetWarmup = func(ctx Context, warmup bool) {
		setWarmupFunc.Call(nil, unsafe.Pointer(&ctx), &warmup)
	}

	vocabBOSFunc, err = currentLib.Prep("llama_vocab_bos", &ffi.TypeSint32, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	VocabBOS = func(vocab Vocab) Token {
		var token ffi.Arg
		vocabBOSFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))

		return Token(token)
	}

	vocabEOSFunc, err = currentLib.Prep("llama_vocab_eos", &ffi.TypeSint32, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	VocabEOS = func(vocab Vocab) Token {
		var token ffi.Arg
		vocabEOSFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))

		return Token(token)
	}

	modelHasEncoderFunc, err = currentLib.Prep("llama_model_has_encoder", &ffi.TypeUint8, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	ModelHasEncoder = func(model Model) bool {
		var result ffi.Arg
		modelHasEncoderFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&model))

		return result.Bool()
	}

	modelHasDecoderFunc, err = currentLib.Prep("llama_model_has_decoder", &ffi.TypeUint8, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	ModelHasDecoder = func(model Model) bool {
		var result ffi.Arg
		modelHasDecoderFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&model))

		return result.Bool()
	}

	encodeFunc, err = currentLib.Prep("llama_encode", &ffi.TypeSint32, &ffi.TypePointer, &TypeBatch)
	if err != nil {
		panic(err)
	}

	Encode = func(ctx Context, batch Batch) int32 {
		var result ffi.Arg
		encodeFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&ctx), unsafe.Pointer(&batch))

		return int32(result)
	}

	batchGetOneFunc, err = currentLib.Prep("llama_batch_get_one", &TypeBatch, &ffi.TypePointer, &ffi.TypeSint32)
	if err != nil {
		panic(err)
	}

	BatchGetOne = func(tokens *Token, nTokens int32) Batch {
		var batch Batch
		batchGetOneFunc.Call(unsafe.Pointer(&batch), unsafe.Pointer(&tokens), unsafe.Pointer(&nTokens))

		return batch
	}

	modelDecoderStartTokenFunc, err = currentLib.Prep("llama_model_decoder_start_token", &ffi.TypeSint32, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	ModelDecoderStartToken = func(model Model) Token {
		var result ffi.Arg
		modelDecoderStartTokenFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&model))

		return Token(result)
	}

	decodeFunc, err = currentLib.Prep("llama_decode", &ffi.TypeSint32, &ffi.TypePointer, &TypeBatch)
	if err != nil {
		panic(err)
	}

	Decode = func(ctx Context, batch Batch) int32 {
		var result ffi.Arg
		decodeFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&ctx), unsafe.Pointer(&batch))

		return int32(result)
	}

	memoryClearFunc, err = currentLib.Prep("llama_memory_clear", &ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeUint8)
	if err != nil {
		panic(err)
	}

	MemoryClear = func(mem Memory, data bool) {
		memoryClearFunc.Call(nil, unsafe.Pointer(&mem), unsafe.Pointer(&data))
	}

	getMemoryFunc, err = currentLib.Prep("llama_get_memory", &ffi.TypePointer, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	GetMemory = func(ctx Context) Memory {
		var mem Memory
		getMemoryFunc.Call(unsafe.Pointer(&mem), unsafe.Pointer(&ctx))

		return mem
	}

	synchronizeFunc, err = currentLib.Prep("llama_synchronize", &ffi.TypeVoid, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	Synchronize = func(ctx Context) {
		synchronizeFunc.Call(nil, unsafe.Pointer(&ctx))
	}

	perfContextResetFunc, err = currentLib.Prep("llama_perf_context_reset", &ffi.TypeVoid, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	PerfContextReset = func(ctx Context) {
		perfContextResetFunc.Call(nil, unsafe.Pointer(&ctx))
	}
}
