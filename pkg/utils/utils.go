package utils

import (
	"unsafe"

	"github.com/wasmvision/go-mtmd/pkg/llama"
)

func Warmup(lctx llama.Context, model llama.Model, vocab llama.Vocab) {
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
