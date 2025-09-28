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
	modelDefaultParamsFunc ffi.Fun

	// LLAMA_API struct llama_model * llama_model_load_from_file(
	//                          const char * path_model,
	//           				struct llama_model_params   params);
	modelLoadFromFileFunc ffi.Fun

	// LLAMA_API struct llama_model_params          llama_model_default_params(void);
	modelFreeFunc ffi.Fun

	// LLAMA_API struct llama_context * llama_init_from_model(
	//                  struct llama_model * model,
	//         			struct llama_context_params   params);
	initFromModelFunc ffi.Fun

	// LLAMA_API const char * llama_model_chat_template(const struct llama_model * model, const char * name);
	modelChatTemplateFunc ffi.Fun

	// LLAMA_API bool llama_model_has_encoder(const struct llama_model * model);
	modelHasEncoderFunc ffi.Fun

	// LLAMA_API bool llama_model_has_decoder(const struct llama_model * model);
	modelHasDecoderFunc ffi.Fun

	// LLAMA_API llama_token llama_model_decoder_start_token(const struct llama_model * model);
	modelDecoderStartTokenFunc ffi.Fun

	// LLAMA_API int32_t llama_model_n_ctx_train(const struct llama_model * model);
	modelNCtxTrainFunc ffi.Fun
)

func loadModelFuncs(lib ffi.Lib) error {
	var err error

	if modelDefaultParamsFunc, err = lib.Prep("llama_model_default_params", &FFITypeModelParams); err != nil {
		return err
	}

	if modelLoadFromFileFunc, err = lib.Prep("llama_model_load_from_file", &ffi.TypePointer, &ffi.TypePointer, &FFITypeModelParams); err != nil {
		return err
	}

	if modelFreeFunc, err = lib.Prep("llama_model_free", &ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return err
	}

	if initFromModelFunc, err = lib.Prep("llama_init_from_model", &ffi.TypePointer, &ffi.TypePointer, &FFITypeContextParams); err != nil {
		return err
	}

	if modelChatTemplateFunc, err = lib.Prep("llama_model_chat_template", &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer); err != nil {
		return err
	}

	if modelHasEncoderFunc, err = lib.Prep("llama_model_has_encoder", &ffi.TypeUint8, &ffi.TypePointer); err != nil {
		return err
	}

	if modelHasDecoderFunc, err = lib.Prep("llama_model_has_decoder", &ffi.TypeUint8, &ffi.TypePointer); err != nil {
		return err
	}

	if modelDecoderStartTokenFunc, err = lib.Prep("llama_model_decoder_start_token", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return err
	}

	if modelNCtxTrainFunc, err = lib.Prep("llama_model_n_ctx_train", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return err
	}

	return nil
}

func ModelDefaultParams() ModelParams {
	var p ModelParams
	modelDefaultParamsFunc.Call(unsafe.Pointer(&p))
	return p
}

func ModelLoadFromFile(pathModel string, params ModelParams) Model {
	var model Model
	file := &[]byte(pathModel + "\x00")[0]
	modelLoadFromFileFunc.Call(unsafe.Pointer(&model), unsafe.Pointer(&file), unsafe.Pointer(&params))
	return model
}

func ModelFree(model Model) {
	modelFreeFunc.Call(nil, unsafe.Pointer(&model))
}

func InitFromModel(model Model, params ContextParams) Context {
	var ctx Context
	initFromModelFunc.Call(unsafe.Pointer(&ctx), unsafe.Pointer(&model), unsafe.Pointer(&params))

	return ctx
}

func ModelChatTemplate(model Model, name string) string {
	var template *byte
	var n *byte
	if len(name) > 0 {
		n = &[]byte(name + "\x00")[0]
	}
	modelChatTemplateFunc.Call(unsafe.Pointer(&template), unsafe.Pointer(&model), unsafe.Pointer(&n))

	return unix.BytePtrToString(template)
}

func ModelHasEncoder(model Model) bool {
	var result ffi.Arg
	modelHasEncoderFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&model))

	return result.Bool()
}

func ModelHasDecoder(model Model) bool {
	var result ffi.Arg
	modelHasDecoderFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&model))

	return result.Bool()
}

func ModelDecoderStartToken(model Model) Token {
	var result ffi.Arg
	modelDecoderStartTokenFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&model))

	return Token(result)
}

func ModelNCtxTrain(model Model) int32 {
	var result ffi.Arg
	modelNCtxTrainFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&model))

	return int32(result)
}
