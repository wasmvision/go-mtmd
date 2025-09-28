package main

import (
	"fmt"

	"github.com/wasmvision/yzma/pkg/llama"
	"github.com/wasmvision/yzma/pkg/loader"
)

var (
	modelFile            = "./models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
	prompt               = "Are you ready to go?"
	libPath              = "./lib"
	responseLength int32 = 32
)

func main() {
	lib, _ := loader.LoadLibrary(libPath)
	llama.Load(lib)

	llama.BackendInit()
	defer llama.BackendFree()

	llama.LogSet(llama.LogSilent(), uintptr(0))

	model := llama.ModelLoadFromFile(modelFile, llama.ModelDefaultParams())
	defer llama.ModelFree(model)

	vocab := llama.ModelGetVocab(model)

	sampler := llama.SamplerChainInit(llama.SamplerChainDefaultParams())
	llama.SamplerChainAdd(sampler, llama.SamplerInitGreedy())

	// call once to get the size
	count := llama.Tokenize(vocab, prompt, nil, true, false)

	// now get the actual tokens
	tokens := make([]llama.Token, count)
	llama.Tokenize(vocab, prompt, tokens, true, false)

	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = uint32(count + responseLength)
	ctxParams.NBatch = uint32(count)

	lctx := llama.InitFromModel(model, ctxParams)
	defer llama.Free(lctx)

	batch := llama.BatchGetOne(tokens)

	if llama.ModelHasEncoder(model) {
		llama.Encode(lctx, batch)

		start := llama.ModelDecoderStartToken(model)
		if start == llama.TOKEN_NULL {
			start = llama.VocabBOS(vocab)
		}

		batch = llama.BatchGetOne([]llama.Token{start})
	}

	for pos := int32(0); pos+batch.NTokens < count+responseLength; pos += batch.NTokens {
		llama.Decode(lctx, batch)

		token := llama.SamplerSample(sampler, lctx, -1)
		llama.SamplerAccept(sampler, token)

		if llama.VocabIsEOG(vocab, token) {
			fmt.Println()
			break
		}

		buf := make([]byte, 36)
		llama.TokenToPiece(vocab, token, buf, 0, true)

		fmt.Print(string(buf))

		batch = llama.BatchGetOne([]llama.Token{token})
	}

	fmt.Println()
}
