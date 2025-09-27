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

	lib, err := loader.LoadLibrary(*libPath)
	if err != nil {
		fmt.Println("unable to load library", err.Error())
		os.Exit(1)
	}
	if err := llama.Load(lib); err != nil {
		fmt.Println("unable to load library", err.Error())
		os.Exit(1)
	}
	if err := mtmd.Load(lib); err != nil {
		fmt.Println("unable to load library", err.Error())
		os.Exit(1)
	}

	llama.BackendInit()
	defer llama.BackendFree()

	fmt.Println("Loading model", *modelFile)
	model := llama.ModelLoadFromFile(*modelFile, llama.ModelDefaultParams())
	defer llama.ModelFree(model)

	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = 4096
	ctxParams.NBatch = 2048

	lctx := llama.InitFromModel(model, ctxParams)
	defer llama.Free(lctx)

	vocab := llama.ModelGetVocab(model)
	sampler := utils.NewSampler(model, utils.DefaultSamplers)
	mtmdCtx := mtmd.InitFromFile(*projFile, model, mtmd.ContextParamsDefault())
	defer mtmd.Free(mtmdCtx)

	output := mtmd.InputChunksInit()
	input := mtmd.NewInputText(chatTemplate(*prompt), true, true)
	bitmap := mtmd.BitmapInitFromFile(mtmdCtx, *imageFile)
	defer mtmd.BitmapFree(bitmap)

	mtmd.Tokenize(mtmdCtx, output, input, []mtmd.Bitmap{bitmap})

	var n llama.Pos
	mtmd.HelperEvalChunks(mtmdCtx, lctx, output, 0, 0, int32(ctxParams.NBatch), true, &n)

	var sz int32 = 1
	batch := llama.BatchInit(1, 0, 1)
	batch.NSeqId = &sz
	batch.NTokens = 1
	seqs := unsafe.SliceData([]llama.SeqId{0})
	batch.SeqId = &seqs

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

		batch.Token = &token
		batch.Pos = &n

		llama.Decode(lctx, batch)
		n++
	}
}

func chatTemplate(prompt string) string {
	chat := []llama.ChatMessage{llama.NewChatMessage("user", prompt+mtmd.DefaultMarker())}
	buf := make([]byte, 1024)

	llama.ChatApplyTemplate("chatml", chat, false, buf)
	result := unix.BytePtrToString(unsafe.SliceData(buf))

	// add to start generation
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
