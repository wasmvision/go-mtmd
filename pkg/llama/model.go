package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
	"golang.org/x/sys/unix"
)

var (
	FFITypeModelParams = ffi.NewType(&ffi.TypePointer, &ffi.TypePointer, &ffi.TypeSint32,
		&ffi.TypeSint32, &ffi.TypeSint32,
		&ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer,
		&ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8)
)

var (
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

	// LLAMA_API struct llama_context * llama_init_from_model(
	//                  struct llama_model * model,
	//         			struct llama_context_params   params);
	InitFromModel     func(model Model, params ContextParams) Context
	initFromModelFunc ffi.Fun

	// LLAMA_API const char * llama_model_chat_template(const struct llama_model * model, const char * name);
	ModelChatTemplate     func(model Model, name string) string
	modelChatTemplateFunc ffi.Fun

	// LLAMA_API bool llama_model_has_encoder(const struct llama_model * model);
	ModelHasEncoder     func(model Model) bool
	modelHasEncoderFunc ffi.Fun

	// LLAMA_API bool llama_model_has_decoder(const struct llama_model * model);
	ModelHasDecoder     func(model Model) bool
	modelHasDecoderFunc ffi.Fun

	// LLAMA_API llama_token llama_model_decoder_start_token(const struct llama_model * model);
	ModelDecoderStartToken     func(model Model) Token
	modelDecoderStartTokenFunc ffi.Fun
)

func initModel(lib ffi.Lib) {
	var err error
	modelDefaultParamsFunc, err = lib.Prep("llama_model_default_params", &FFITypeModelParams)
	if err != nil {
		panic(err)
	}

	ModelDefaultParams = func() ModelParams {
		var p ModelParams
		modelDefaultParamsFunc.Call(unsafe.Pointer(&p))
		return p
	}

	modelLoadFromFileFunc, err = lib.Prep("llama_model_load_from_file", &ffi.TypePointer, &ffi.TypePointer, &FFITypeModelParams)
	if err != nil {
		panic(err)
	}

	ModelLoadFromFile = func(pathModel string, params ModelParams) Model {
		var model Model
		file := &[]byte(pathModel + "\x00")[0]
		modelLoadFromFileFunc.Call(unsafe.Pointer(&model), unsafe.Pointer(&file), unsafe.Pointer(&params))
		return model
	}

	modelFreeFunc, err = lib.Prep("llama_model_free", &ffi.TypeVoid, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	ModelFree = func(model Model) {
		modelFreeFunc.Call(nil, unsafe.Pointer(&model))
	}

	initFromModelFunc, err = lib.Prep("llama_init_from_model", &ffi.TypePointer, &ffi.TypePointer, &FFITypeContextParams)
	if err != nil {
		panic(err)
	}

	InitFromModel = func(model Model, params ContextParams) Context {
		var ctx Context
		initFromModelFunc.Call(unsafe.Pointer(&ctx), unsafe.Pointer(&model), unsafe.Pointer(&params))

		return ctx
	}

	modelChatTemplateFunc, err = lib.Prep("llama_model_chat_template", &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	ModelChatTemplate = func(model Model, name string) string {
		var template *byte
		n := &[]byte(name + "\x00")[0]
		modelChatTemplateFunc.Call(unsafe.Pointer(&template), unsafe.Pointer(&model), unsafe.Pointer(&n))

		return unix.BytePtrToString(template)
	}

	modelHasEncoderFunc, err = lib.Prep("llama_model_has_encoder", &ffi.TypeUint8, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	ModelHasEncoder = func(model Model) bool {
		var result ffi.Arg
		modelHasEncoderFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&model))

		return result.Bool()
	}

	modelHasDecoderFunc, err = lib.Prep("llama_model_has_decoder", &ffi.TypeUint8, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	ModelHasDecoder = func(model Model) bool {
		var result ffi.Arg
		modelHasDecoderFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&model))

		return result.Bool()
	}

	modelDecoderStartTokenFunc, err = lib.Prep("llama_model_decoder_start_token", &ffi.TypeSint32, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	ModelDecoderStartToken = func(model Model) Token {
		var result ffi.Arg
		modelDecoderStartTokenFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&model))

		return Token(result)
	}

}
