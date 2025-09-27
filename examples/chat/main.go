package main

import (
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/wasmvision/yzma/pkg/llama"
	"github.com/wasmvision/yzma/pkg/loader"
)

var (
	modelFile *string
	prompt    *string
	libPath   *string
	verbose   *bool
)

func main() {
	if err := handleFlags(); err != nil {
		showUsage()
		os.Exit(0)
	}

	lib := loader.LoadLibrary(*libPath)
	llama.Load(lib)

	llama.BackendInit()
	defer llama.BackendFree()

	if !*verbose {
		llama.LogSet(llama.LogSilent(), uintptr(0))
	}

	model := llama.ModelLoadFromFile(*modelFile, llama.ModelDefaultParams())
	defer llama.ModelFree(model)

	vocab := llama.ModelGetVocab(model)

	sampler := llama.SamplerChainInit(llama.SamplerChainDefaultParams())
	llama.SamplerChainAdd(sampler, llama.SamplerInitGreedy())

	// call once to get the size
	count := llama.Tokenize(vocab, *prompt, nil, true, false)

	// now get the actual tokens
	tokens := make([]llama.Token, count)
	llama.Tokenize(vocab, *prompt, tokens, true, false)

	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = uint32(count + 32)
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

	fmt.Println()

	for pos := int32(0); pos+batch.NTokens < count+32; pos += batch.NTokens {
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

func showUsage() {
	fmt.Println(`
Usage:
chat -model [model file path] -lib [llama.cpp .so file path] -prompt [what you want to ask] -v`)
}

func handleFlags() error {
	modelFile = flag.String("model", "", "model file to use")
	prompt = flag.String("prompt", "", "prompt")
	libPath = flag.String("lib", "", "path to llama.cpp compiled library files")
	verbose = flag.Bool("v", false, "verbose logging")

	flag.Parse()

	if len(*modelFile) == 0 ||
		len(*prompt) == 0 ||
		len(*libPath) == 0 {

		return errors.New("missing a flag")
	}

	return nil
}
