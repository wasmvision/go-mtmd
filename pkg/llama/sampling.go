package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
)

type Sampler uintptr

var (
	TypeSamplerChainParams = ffi.NewType(&ffi.TypePointer)
)

var (
	// LLAMA_API struct llama_sampler_chain_params  llama_sampler_chain_default_params(void);
	SamplerChainDefaultParams     func() SamplerChainParams
	samplerChainDefaultParamsFunc ffi.Fun

	// LLAMA_API struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params);
	SamplerChainInit     func(params SamplerChainParams) Sampler
	samplerChainInitFunc ffi.Fun

	// LLAMA_API void llama_sampler_chain_add(struct llama_sampler * chain, struct llama_sampler * smpl);
	SamplerChainAdd     func(chain Sampler, smpl Sampler)
	samplerChainAddFunc ffi.Fun

	// LLAMA_API struct llama_sampler * llama_sampler_init_greedy(void);
	SamplerInitGreedy     func() Sampler
	samplerInitGreedyFunc ffi.Fun

	// LLAMA_API struct llama_sampler * llama_sampler_init_dist  (uint32_t seed);
	SamplerInitDist     func(seed uint32) Sampler
	samplerInitDistFunc ffi.Fun

	// LLAMA_API struct llama_sampler * llama_sampler_init_logit_bias(
	//                  int32_t   n_vocab,
	//                  int32_t   n_logit_bias,
	//   				const llama_logit_bias * logit_bias);
	SamplerInitLogitBias     func(nVocab int32, nLogitBias int32, logitBias *LogitBias) Sampler
	samplerInitLogitBiasFunc ffi.Fun

	// LLAMA_API llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);
	SamplerSample     func(smpl Sampler, ctx Context, idx int32) Token
	samplerSampleFunc ffi.Fun

	// LLAMA_API void  llama_sampler_accept(struct llama_sampler * smpl, llama_token token);
	SamplerAccept     func(smpl Sampler, token Token)
	samplerAcceptFunc ffi.Fun

	// LLAMA_API void lama_sampler_free  (struct llama_sampler * smpl);
	SamplerFree     func(smpl Sampler)
	samplerFreeFunc ffi.Fun
)

func initSampling(lib ffi.Lib) {
	var err error
	samplerChainDefaultParamsFunc, err = lib.Prep("llama_sampler_chain_default_params", &TypeSamplerChainParams)
	if err != nil {
		panic(err)
	}

	SamplerChainDefaultParams = func() SamplerChainParams {
		var p SamplerChainParams
		samplerChainDefaultParamsFunc.Call(unsafe.Pointer(&p))

		return p
	}

	samplerChainInitFunc, err = lib.Prep("llama_sampler_chain_init", &ffi.TypePointer, &TypeSamplerChainParams)
	if err != nil {
		panic(err)
	}

	SamplerChainInit = func(params SamplerChainParams) Sampler {
		var p Sampler
		samplerChainInitFunc.Call(unsafe.Pointer(&p), unsafe.Pointer(&params))

		return p
	}

	samplerChainAddFunc, err = lib.Prep("llama_sampler_chain_add", &ffi.TypeVoid, &ffi.TypePointer, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	SamplerChainAdd = func(chain Sampler, smpl Sampler) {
		samplerChainAddFunc.Call(nil, unsafe.Pointer(&chain), unsafe.Pointer(&smpl))
	}

	samplerInitGreedyFunc, err = lib.Prep("llama_sampler_init_greedy", &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	SamplerInitGreedy = func() Sampler {
		var p Sampler
		samplerInitGreedyFunc.Call(unsafe.Pointer(&p))

		return p
	}

	samplerInitDistFunc, err = lib.Prep("llama_sampler_init_dist", &ffi.TypePointer, &ffi.TypeUint32)
	if err != nil {
		panic(err)
	}

	SamplerInitDist = func(seed uint32) Sampler {
		var p Sampler
		samplerInitDistFunc.Call(unsafe.Pointer(&p), unsafe.Pointer(&seed))

		return p
	}

	samplerInitLogitBiasFunc, err = lib.Prep("llama_sampler_init_logit_bias", &ffi.TypePointer, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	SamplerInitLogitBias = func(nVocab int32, nLogitBias int32, logitBias *LogitBias) Sampler {
		var p Sampler
		samplerInitLogitBiasFunc.Call(unsafe.Pointer(&p), unsafe.Pointer(&nVocab), unsafe.Pointer(&nLogitBias), unsafe.Pointer(&logitBias))

		return p
	}

	samplerSampleFunc, err = lib.Prep("llama_sampler_sample", &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypeSint32)
	if err != nil {
		panic(err)
	}

	SamplerSample = func(smpl Sampler, ctx Context, idx int32) Token {
		var result ffi.Arg
		samplerSampleFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&smpl), unsafe.Pointer(&ctx), unsafe.Pointer(&idx))

		return Token(result)
	}

	samplerAcceptFunc, err = lib.Prep("llama_sampler_accept", &ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeSint32)
	if err != nil {
		panic(err)
	}

	SamplerAccept = func(smpl Sampler, token Token) {
		samplerAcceptFunc.Call(nil, unsafe.Pointer(&smpl), unsafe.Pointer(&token))
	}

	samplerFreeFunc, err = lib.Prep("llama_sampler_free", &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	SamplerFree = func(smpl Sampler) {
		samplerFreeFunc.Call(nil, unsafe.Pointer(&smpl))
	}
}
