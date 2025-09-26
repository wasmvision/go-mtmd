package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"unsafe"

	"github.com/wasmvision/yzma/pkg/llama"
	"github.com/wasmvision/yzma/pkg/loader"
	"golang.org/x/sys/unix"
)

var (
	modelFile *string
	prompt    *string
	libPath   *string
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

	params := llama.ModelDefaultParams()
	model := llama.ModelLoadFromFile(*modelFile, params)
	defer llama.ModelFree(model)

	vocab := llama.ModelGetVocab(model)
	sampler := setupSampler(model, vocab)

	p, _ := unix.BytePtrFromString(*prompt)
	count := llama.Tokenize(vocab, p, int32(len(*prompt)), nil, 0, true, false)

	tokens := make([]llama.Token, -count)
	llama.Tokenize(vocab, p, int32(len(*prompt)), unsafe.SliceData(tokens), -count, true, false)

	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = uint32(-count + 32)
	ctxParams.NBatch = uint32(-count)

	lctx := llama.InitFromModel(model, ctxParams)
	defer llama.Free(lctx)

	batch := llama.BatchGetOne(unsafe.SliceData(tokens), int32(len(tokens)))

	if llama.ModelHasEncoder(model) {
		llama.Encode(lctx, batch)

		start := llama.ModelDecoderStartToken(model)
		if start == llama.TOKEN_NULL {
			start = llama.VocabBOS(vocab)
		}

		batch = llama.BatchGetOne(unsafe.SliceData([]llama.Token{start}), int32(1))
	}

	fmt.Println()

	for pos := int32(0); pos+batch.NTokens < count+32; pos += batch.NTokens {
		llama.Decode(lctx, batch)

		token := llama.SamplerSample(sampler, lctx, -1)
		llama.SamplerAccept(sampler, token)

		if llama.VocabIsEOG(vocab, token) {
			// end of generation
			fmt.Println()
			break
		}

		data := make([]byte, 36)
		buf := unsafe.SliceData(data)
		llama.TokenToPiece(vocab, token, buf, 36, 0, true)

		res := unix.BytePtrToString(buf)
		fmt.Print(res)

		batch = llama.BatchGetOne(unsafe.SliceData([]llama.Token{token}), int32(1))
	}

	fmt.Println()
}

func setupSampler(model llama.Model, vocab llama.Vocab) llama.Sampler {
	params := llama.SamplerChainDefaultParams()
	sampler := llama.SamplerChainInit(params)

	llama.SamplerChainAdd(sampler, llama.SamplerInitGreedy())

	return sampler
}

func showUsage() {
	fmt.Println(`
Usage:
chat -model [model file path] -lib [llama.cpp .so file path] -prompt [what you want to ask]`)
}

func handleFlags() error {
	modelFile = flag.String("model", "", "model file to use")
	prompt = flag.String("prompt", "", "prompt")
	libPath = flag.String("lib", "", "path to llama.cpp compiled library files")

	flag.Parse()

	if len(*modelFile) == 0 ||
		len(*prompt) == 0 ||
		len(*libPath) == 0 {

		return errors.New("missing a flag")
	}

	return nil
}
