package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"unsafe"

	"github.com/wasmvision/yzma/pkg/llama"
	"github.com/wasmvision/yzma/pkg/loader"
	"github.com/wasmvision/yzma/pkg/mtmd"
	"github.com/wasmvision/yzma/pkg/utils"
	"golang.org/x/sys/unix"
)

var (
	modelFile *string
	projFile  *string
	prompt    *string
	imageFile *string
	libPath   *string
)

func main() {
	if err := handleFlags(); err != nil {
		showUsage()
		os.Exit(0)
	}

	lib := loader.LoadLibrary(*libPath)
	llama.Load(lib)
	mtmd.Load(lib)

	llama.BackendInit()
	defer llama.BackendFree()

	params := llama.ModelDefaultParams()

	fmt.Println("Loading model", *modelFile)
	model := llama.ModelLoadFromFile(*modelFile, params)
	defer llama.ModelFree(model)

	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = 4096
	ctxParams.NBatch = 2048

	lctx := llama.InitFromModel(model, ctxParams)
	defer llama.Free(lctx)

	vocab := llama.ModelGetVocab(model)

	// add default samplers
	sampler := utils.NewSampler(model, []llama.SamplerType{
		llama.SamplerTypePenalties,
		llama.SamplerTypeDry,
		llama.SamplerTypeTopNSigma,
		llama.SamplerTypeTopK,
		llama.SamplerTypeTypicalP,
		llama.SamplerTypeTopP,
		llama.SamplerTypeMinP,
		llama.SamplerTypeXTC,
		llama.SamplerTypeTemperature,
	})

	// fmt.Println("warming up")
	// utils.Warmup(lctx, model, vocab)

	mctxParams := mtmd.ContextParamsDefault()
	// mctxParams.UseGPU = true
	// mctxParams.Verbosity = llama.LogLevel(1)

	mtmdCtx := mtmd.InitFromFile(*projFile, model, mctxParams)
	defer mtmd.Free(mtmdCtx)

	input := mtmd.NewInputText(chatTemplate(*prompt), true, true)
	output := mtmd.InputChunksInit()

	bitmap := mtmd.BitmapInitFromFile(mtmdCtx, *imageFile)
	defer mtmd.BitmapFree(bitmap)

	mtmd.Tokenize(mtmdCtx, output, input, []mtmd.Bitmap{bitmap})

	var n llama.Pos
	mtmd.HelperEvalChunks(mtmdCtx, lctx, output, 0, 0, int32(ctxParams.NBatch), true, &n)

	batch := llama.BatchInit(1, 0, 1)

	fmt.Println()

	for i := 0; i < llama.MaxToken; i++ {
		token := llama.SamplerSample(sampler, lctx, -1)
		llama.SamplerAccept(sampler, token)

		if llama.VocabIsEOG(vocab, token) {
			fmt.Println()
			break
		}

		buf := make([]byte, 128)
		llama.TokenToPiece(vocab, token, buf, 0, true)

		fmt.Print(string(buf))

		batch.NTokens = 1
		batch.Token = &token
		batch.Pos = &n
		var sz int32 = 1
		batch.NSeqId = &sz
		seqs := unsafe.SliceData([]llama.SeqId{0})
		batch.SeqId = &seqs

		llama.Decode(lctx, batch)
		n++
	}
}

func chatTemplate(prompt string) string {
	role, _ := unix.BytePtrFromString("user")
	content, _ := unix.BytePtrFromString(prompt + mtmd.DefaultMarker())

	chat := []llama.ChatMessage{llama.ChatMessage{Role: role, Content: content}}
	buf := make([]byte, 1024)

	llama.ChatApplyTemplate("chatml", chat, false, buf)
	result := unix.BytePtrToString(unsafe.SliceData(buf))

	// start generation
	result += "<|im_start|>assistant\n"

	return result
}

func showUsage() {
	fmt.Println(`
Usage:
vlm -model [model file path] -proj [projector file path] -lib [llama.cpp .so file path] -prompt [what you want to ask] -image [image file path]`)
}

func handleFlags() error {
	modelFile = flag.String("model", "", "model file to use")
	projFile = flag.String("proj", "", "projector file to use")
	prompt = flag.String("prompt", "what is this?", "prompt")
	imageFile = flag.String("image", "", "image file to use")
	libPath = flag.String("lib", "", "path to llama.cpp compiled library files")

	flag.Parse()

	if len(*modelFile) == 0 ||
		len(*projFile) == 0 ||
		len(*prompt) == 0 ||
		len(*imageFile) == 0 ||
		len(*libPath) == 0 {

		return errors.New("missing a flag")
	}

	return nil
}
