package main

import (
	"fmt"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/loader"
)

var (
	modelFile            = "./models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
	prompt               = "Are you ready to rock?"
	libPath              = "./lib"
	responseLength int32 = 64
)

func main() {
	lib, _ := loader.LoadLibrary(libPath)
	llama.Load(lib)

	llama.BackendInit()
	llama.LogSet(llama.LogSilent(), uintptr(0))

	model := llama.ModelLoadFromFile(modelFile, llama.ModelDefaultParams())
	vocab := llama.ModelGetVocab(model)

	sampler := llama.SamplerChainInit(llama.SamplerChainDefaultParams())
	llama.SamplerChainAdd(sampler, llama.SamplerInitGreedy())

	// call once to get the size of the tokens from the prompt
	count := llama.Tokenize(vocab, prompt, nil, true, false)

	// now get the actual tokens
	tokens := make([]llama.Token, count)
	llama.Tokenize(vocab, prompt, tokens, true, false)

	lctx := llama.InitFromModel(model, llama.ContextDefaultParams())
	batch := llama.BatchGetOne(tokens)
	for pos := int32(0); pos+batch.NTokens < count+responseLength; pos += batch.NTokens {
		llama.Decode(lctx, batch)
		token := llama.SamplerSample(sampler, lctx, -1)

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
