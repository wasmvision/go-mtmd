package utils

import (
	"math"
	"unsafe"

	"github.com/wasmvision/yzma/pkg/llama"
	"golang.org/x/sys/unix"
)

// NewSampler creates a new sampling chain.
func NewSampler(model llama.Model, samplers []llama.SamplerType) llama.Sampler {
	vocab := llama.ModelGetVocab(model)
	nTokens := llama.VocabNTokens(vocab)

	params := llama.SamplerChainDefaultParams()
	sampler := llama.SamplerChainInit(params)

	logitBiasEOG := make([]llama.LogitBias, 0)

	for i := int32(0); i < nTokens; i++ {
		token := llama.Token(i)
		if llama.VocabIsEOG(vocab, token) {
			logitBiasEOG = append(logitBiasEOG, llama.LogitBias{Token: token, Bias: math.SmallestNonzeroFloat32})
		}
	}

	bias := llama.SamplerInitLogitBias(nTokens, int32(len(logitBiasEOG)), unsafe.SliceData(logitBiasEOG))
	llama.SamplerChainAdd(sampler, bias)

	for samplerType := range samplers {
		switch samplerType {
		case llama.SamplerTypeDry:
			seqBreakers := []string{"\n", ":", "\"", "*"}
			var combined []*byte
			for _, s := range seqBreakers {
				ptr, err := unix.BytePtrFromString(s)
				if err != nil {
					panic(err)
				}
				combined = append(combined, ptr)
			}
			seqBreakersPtr := unsafe.SliceData(combined)

			dry := llama.SamplerInitDry(vocab, llama.ModelNCtxTrain(model), 0, 1.75, 2, 4096, seqBreakersPtr, uint32(len(seqBreakers)))
			llama.SamplerChainAdd(sampler, dry)

		case llama.SamplerTypeTopK:
			topK := llama.SamplerInitTopK(40)
			llama.SamplerChainAdd(sampler, topK)

		case llama.SamplerTypeTopP:
			topP := llama.SamplerInitTopP(0.95, 0)
			llama.SamplerChainAdd(sampler, topP)

		case llama.SamplerTypeMinP:
			minP := llama.SamplerInitMinP(0.05, 0)
			llama.SamplerChainAdd(sampler, minP)

		case llama.SamplerTypeTypicalP:
			typical := llama.SamplerInitTypical(1.0, 0)
			llama.SamplerChainAdd(sampler, typical)

		case llama.SamplerTypeTemperature:
			temp := llama.SamplerInitTempExt(0.2, 0, 1.0)
			llama.SamplerChainAdd(sampler, temp)

		case llama.SamplerTypeXTC:
			xtc := llama.SamplerInitXTC(0, 0.1, 0, llama.DEFAULT_SEED)
			llama.SamplerChainAdd(sampler, xtc)

		case llama.SamplerTypeInfill:
			// TODO: add implementation

		case llama.SamplerTypePenalties:
			penalties := llama.SamplerInitPenalties(64, 1.0, 0, 0)
			llama.SamplerChainAdd(sampler, penalties)

		case llama.SamplerTypeTopNSigma:
			topNSigma := llama.SamplerInitTopNSigma(-1.0)
			llama.SamplerChainAdd(sampler, topNSigma)
		}
	}

	// always add this last
	dist := llama.SamplerInitDist(llama.DEFAULT_SEED)
	llama.SamplerChainAdd(sampler, dist)

	return sampler
}

// Warmup is to warm-up a model.
func Warmup(lctx llama.Context, model llama.Model) {
	vocab := llama.ModelGetVocab(model)

	llama.SetWarmup(lctx, true)

	tokens := make([]llama.Token, 0)
	bos := llama.VocabBOS(vocab)
	eos := llama.VocabEOS(vocab)

	if bos != llama.TOKEN_NULL {
		tokens = append(tokens, bos)
	}
	if eos != llama.TOKEN_NULL {
		tokens = append(tokens, eos)
	}
	if len(tokens) == 0 {
		tokens = append(tokens, 0)
	}

	if llama.ModelHasEncoder(model) {
		batch := llama.BatchGetOne(unsafe.SliceData(tokens), int32(len(tokens)))
		llama.Encode(lctx, batch)

		start := llama.ModelDecoderStartToken(model)
		if start == llama.TOKEN_NULL {
			start = bos
		}
		tokens = append([]llama.Token{}, start)
	}

	if llama.ModelHasDecoder(model) {
		batch := llama.BatchGetOne(unsafe.SliceData(tokens), int32(len(tokens)))
		llama.Decode(lctx, batch)
	}

	mem := llama.GetMemory(lctx)
	llama.MemoryClear(mem, true)

	llama.Synchronize(lctx)

	// llama_perf_context_reset(lctx);
	llama.SetWarmup(lctx, false)
}
