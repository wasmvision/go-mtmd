package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
)

type SamplerType int32

const (
	SamplerTypeNone        SamplerType = iota
	SamplerTypeDry                     = 1
	SamplerTypeTopK                    = 2
	SamplerTypeTopP                    = 3
	SamplerTypeMinP                    = 4
	SamplerTypeTypicalP                = 6
	SamplerTypeTemperature             = 7
	SamplerTypeXTC                     = 8
	SamplerTypeInfill                  = 9
	SamplerTypePenalties               = 10
	SamplerTypeTopNSigma               = 11
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

	// LLAMA_API struct llama_sampler * llama_sampler_init_penalties(
	// 						int32_t   penalty_last_n,   // last n tokens to penalize (0 = disable penalty, -1 = context size)
	// 						float   penalty_repeat,   // 1.0 = disabled
	// 						float   penalty_freq,     // 0.0 = disabled
	// 						float   penalty_present); // 0.0 = disabled
	SamplerInitPenalties     func(lastN int32, repeat float32, freq float32, present float32) Sampler
	samplerInitPenaltiesFunc ffi.Fun

	// LLAMA_API struct llama_sampler * llama_sampler_init_dry(
	// 	const struct llama_vocab *  vocab,
	// 						int32_t    n_ctx_train,
	// 						float    dry_multiplier,
	// 						float    dry_base,
	// 						int32_t    dry_allowed_length,
	// 						int32_t    dry_penalty_last_n,
	// 					const char ** seq_breakers,
	// 						size_t    num_breakers);
	SamplerInitDry func(vocab Vocab, nCtxTrain int32, multiplier float32, base float32, allowedLength int32, penaltyLast int32,
		seqBreakers **byte, numBreakers uint32) Sampler
	samplerInitDryFunc ffi.Fun

	// LLAMA_API struct llama_sampler * llama_sampler_init_top_n_sigma(float   n);
	SamplerInitTopNSigma     func(n float32) Sampler
	samplerInitTopNSigmaFunc ffi.Fun

	// LLAMA_API struct llama_sampler * llama_sampler_init_top_k      (int32_t k);
	SamplerInitTopK     func(k int32) Sampler
	samplerInitTopKFunc ffi.Fun

	// LLAMA_API struct llama_sampler * llama_sampler_init_typical    (float   p, size_t min_keep);
	SamplerInitTypical     func(p float32, keep uint32) Sampler
	samplerInitTypicalFunc ffi.Fun

	// LLAMA_API struct llama_sampler * llama_sampler_init_top_p      (float   p, size_t min_keep);
	SamplerInitTopP     func(p float32, keep uint32) Sampler
	samplerInitTopPFunc ffi.Fun

	// LLAMA_API struct llama_sampler * llama_sampler_init_min_p      (float   p, size_t min_keep);
	SamplerInitMinP     func(p float32, keep uint32) Sampler
	samplerInitMinPFunc ffi.Fun

	// LLAMA_API struct llama_sampler * llama_sampler_init_xtc        (float   p, float   t,     size_t min_keep, uint32_t seed);
	SamplerInitXTC     func(p float32, t float32, minKeep uint32, seed uint32) Sampler
	samplerInitXTCFunc ffi.Fun

	// LLAMA_API struct llama_sampler * llama_sampler_init_temp_ext   (float   t, float   delta, float exponent);
	SamplerInitTempExt     func(t float32, delta float32, exponent float32) Sampler
	samplerInitTempExtFunc ffi.Fun

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

func loadSamplingFuncs(lib ffi.Lib) {
	var err error
	if samplerChainDefaultParamsFunc, err = lib.Prep("llama_sampler_chain_default_params", &TypeSamplerChainParams); err != nil {
		panic(err)
	}
	SamplerChainDefaultParams = func() SamplerChainParams {
		var p SamplerChainParams
		samplerChainDefaultParamsFunc.Call(unsafe.Pointer(&p))

		return p
	}

	if samplerChainInitFunc, err = lib.Prep("llama_sampler_chain_init", &ffi.TypePointer, &TypeSamplerChainParams); err != nil {
		panic(err)
	}
	SamplerChainInit = func(params SamplerChainParams) Sampler {
		var p Sampler
		samplerChainInitFunc.Call(unsafe.Pointer(&p), unsafe.Pointer(&params))

		return p
	}

	if samplerChainAddFunc, err = lib.Prep("llama_sampler_chain_add", &ffi.TypeVoid, &ffi.TypePointer, &ffi.TypePointer); err != nil {
		panic(err)
	}
	SamplerChainAdd = func(chain Sampler, smpl Sampler) {
		samplerChainAddFunc.Call(nil, unsafe.Pointer(&chain), unsafe.Pointer(&smpl))
	}

	if samplerInitGreedyFunc, err = lib.Prep("llama_sampler_init_greedy", &ffi.TypePointer); err != nil {
		panic(err)
	}
	SamplerInitGreedy = func() Sampler {
		var p Sampler
		samplerInitGreedyFunc.Call(unsafe.Pointer(&p))

		return p
	}

	if samplerInitDistFunc, err = lib.Prep("llama_sampler_init_dist", &ffi.TypePointer, &ffi.TypeUint32); err != nil {
		panic(err)
	}
	SamplerInitDist = func(seed uint32) Sampler {
		var p Sampler
		samplerInitDistFunc.Call(unsafe.Pointer(&p), unsafe.Pointer(&seed))

		return p
	}

	if samplerInitLogitBiasFunc, err = lib.Prep("llama_sampler_init_logit_bias", &ffi.TypePointer, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		panic(err)
	}
	SamplerInitLogitBias = func(nVocab int32, nLogitBias int32, logitBias *LogitBias) Sampler {
		var p Sampler
		samplerInitLogitBiasFunc.Call(unsafe.Pointer(&p), unsafe.Pointer(&nVocab), unsafe.Pointer(&nLogitBias), unsafe.Pointer(&logitBias))

		return p
	}

	if samplerInitPenaltiesFunc, err = lib.Prep("llama_sampler_init_penalties", &ffi.TypePointer, &ffi.TypeSint32, &ffi.TypeFloat, &ffi.TypeFloat, &ffi.TypeFloat); err != nil {
		panic(err)
	}
	SamplerInitPenalties = func(lastN int32, repeat float32, freq float32, present float32) Sampler {
		var p Sampler
		samplerInitPenaltiesFunc.Call(unsafe.Pointer(&p), unsafe.Pointer(&lastN), unsafe.Pointer(&repeat), unsafe.Pointer(&freq), unsafe.Pointer(&present))

		return p
	}

	if samplerInitDryFunc, err = lib.Prep("llama_sampler_init_dry", &ffi.TypePointer, &ffi.TypeSint32, &ffi.TypeFloat, &ffi.TypeFloat,
		&ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeUint32); err != nil {
		panic(err)
	}
	SamplerInitDry = func(vocab Vocab, nCtxTrain int32, multiplier float32, base float32, allowedLength int32, penaltyLast int32,
		seqBreakers **byte, numBreakers uint32) Sampler {
		var p Sampler
		samplerInitDryFunc.Call(unsafe.Pointer(&p), unsafe.Pointer(&nCtxTrain), unsafe.Pointer(&multiplier), unsafe.Pointer(&base), unsafe.Pointer(&allowedLength), unsafe.Pointer(&penaltyLast),
			unsafe.Pointer(seqBreakers), unsafe.Pointer(&numBreakers))

		return p
	}

	if samplerInitTopNSigmaFunc, err = lib.Prep("llama_sampler_init_top_n_sigma", &ffi.TypePointer, &ffi.TypeFloat); err != nil {
		panic(err)
	}
	SamplerInitTopNSigma = func(n float32) Sampler {
		var p Sampler
		samplerInitTopNSigmaFunc.Call(unsafe.Pointer(&p), unsafe.Pointer(&n))

		return p
	}

	if samplerInitTopKFunc, err = lib.Prep("llama_sampler_init_top_k", &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		panic(err)
	}
	SamplerInitTopK = func(k int32) Sampler {
		var p Sampler
		samplerInitTopKFunc.Call(unsafe.Pointer(&p), unsafe.Pointer(&k))

		return p
	}

	if samplerInitTypicalFunc, err = lib.Prep("llama_sampler_init_typical", &ffi.TypePointer, &ffi.TypeFloat, &ffi.TypeUint32); err != nil {
		panic(err)
	}
	SamplerInitTypical = func(p float32, keep uint32) Sampler {
		var s Sampler
		samplerInitTypicalFunc.Call(unsafe.Pointer(&s), unsafe.Pointer(&p), unsafe.Pointer(&keep))

		return s
	}

	if samplerInitTopPFunc, err = lib.Prep("llama_sampler_init_top_p", &ffi.TypePointer, &ffi.TypeFloat, &ffi.TypeUint32); err != nil {
		panic(err)
	}
	SamplerInitTopP = func(p float32, keep uint32) Sampler {
		var s Sampler
		samplerInitTopPFunc.Call(unsafe.Pointer(&s), unsafe.Pointer(&p), unsafe.Pointer(&keep))

		return s
	}

	if samplerInitMinPFunc, err = lib.Prep("llama_sampler_init_top_p", &ffi.TypePointer, &ffi.TypeFloat, &ffi.TypeUint32); err != nil {
		panic(err)
	}
	SamplerInitMinP = func(p float32, keep uint32) Sampler {
		var s Sampler
		samplerInitMinPFunc.Call(unsafe.Pointer(&s), unsafe.Pointer(&p), unsafe.Pointer(&keep))

		return s
	}

	if samplerInitXTCFunc, err = lib.Prep("llama_sampler_init_xtc", &ffi.TypePointer, &ffi.TypeFloat, &ffi.TypeFloat, &ffi.TypeUint32, &ffi.TypeUint32); err != nil {
		panic(err)
	}
	SamplerInitXTC = func(p float32, t float32, minKeep uint32, seed uint32) Sampler {
		var s Sampler
		samplerInitXTCFunc.Call(unsafe.Pointer(&s), unsafe.Pointer(&p), unsafe.Pointer(&t), unsafe.Pointer(&minKeep), unsafe.Pointer(&seed))

		return s
	}

	if samplerInitTempExtFunc, err = lib.Prep("llama_sampler_init_temp_ext", &ffi.TypePointer, &ffi.TypeFloat, &ffi.TypeFloat, &ffi.TypeFloat); err != nil {
		panic(err)
	}
	SamplerInitTempExt = func(t float32, delta float32, exponent float32) Sampler {
		var s Sampler
		samplerInitTempExtFunc.Call(unsafe.Pointer(&s), unsafe.Pointer(&t), unsafe.Pointer(&delta), unsafe.Pointer(&exponent))

		return s
	}

	if samplerSampleFunc, err = lib.Prep("llama_sampler_sample", &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		panic(err)
	}
	SamplerSample = func(smpl Sampler, ctx Context, idx int32) Token {
		var result ffi.Arg
		samplerSampleFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&smpl), unsafe.Pointer(&ctx), unsafe.Pointer(&idx))

		return Token(result)
	}

	if samplerAcceptFunc, err = lib.Prep("llama_sampler_accept", &ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		panic(err)
	}
	SamplerAccept = func(smpl Sampler, token Token) {
		samplerAcceptFunc.Call(nil, unsafe.Pointer(&smpl), unsafe.Pointer(&token))
	}

	if samplerFreeFunc, err = lib.Prep("llama_sampler_free", &ffi.TypePointer); err != nil {
		panic(err)
	}
	SamplerFree = func(smpl Sampler) {
		samplerFreeFunc.Call(nil, unsafe.Pointer(&smpl))
	}
}
