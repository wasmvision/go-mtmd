package main

import (
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"unsafe"

	"github.com/wasmvision/yzma/pkg/llama"
	"github.com/wasmvision/yzma/pkg/loader"
	"github.com/wasmvision/yzma/pkg/mtmd"
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
	llama.Init(lib)
	mtmd.Init(lib)

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
	sampler := setupSampler(model, vocab)

	// fmt.Println("warming up")
	// utils.Warmup(lctx, model, vocab)

	mctxParams := mtmd.ContextParamsDefault()
	// mctxParams.UseGPU = true
	// mctxParams.Verbosity = llama.LogLevel(1)

	mtmdCtx := mtmd.InitFromFile(*projFile, model, mctxParams)
	defer mtmd.Free(mtmdCtx)

	pmpt := chatTemplate(*prompt)
	p, _ := unix.BytePtrFromString(pmpt)
	input := &mtmd.InputText{
		Text:         p,
		AddSpecial:   true,
		ParseSpecial: true,
	}

	output := mtmd.InputChunksInit()

	bitmap := mtmd.BitmapInitFromFile(mtmdCtx, *imageFile)
	defer mtmd.BitmapFree(bitmap)

	bitmaps := []mtmd.Bitmap{bitmap}
	bt := unsafe.SliceData(bitmaps)
	mtmd.Tokenize(mtmdCtx, output, input, bt, 1)

	var n llama.Pos
	mtmd.HelperEvalChunks(mtmdCtx, lctx, output, 0, 0, int32(ctxParams.NBatch), true, &n)

	batch := llama.BatchInit(1, 0, 1)

	fmt.Println()

	for i := 0; i < llama.MaxToken; i++ {
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
	return fmt.Sprintf("<|im_start|>user\n%s<__media__><|im_end|>\n<|im_start|>assistant\n", prompt)
}

func setupSampler(model llama.Model, vocab llama.Vocab) llama.Sampler {
	params := llama.SamplerChainDefaultParams()
	sampler := llama.SamplerChainInit(params)

	logitBiasEOG := make([]llama.LogitBias, 0)
	nTokens := llama.VocabNTokens(vocab)

	for i := int32(0); i < nTokens; i++ {
		token := llama.Token(i)
		if llama.VocabIsEOG(vocab, token) {
			logitBiasEOG = append(logitBiasEOG, llama.LogitBias{Token: token, Bias: math.SmallestNonzeroFloat32})
		}
	}

	bias := llama.SamplerInitLogitBias(nTokens, int32(len(logitBiasEOG)), unsafe.SliceData(logitBiasEOG))
	llama.SamplerChainAdd(sampler, bias)

	// defaults: penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature
	penalties := llama.SamplerInitPenalties(64, 1.0, 0, 0)
	llama.SamplerChainAdd(sampler, penalties)

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

	topNSigma := llama.SamplerInitTopNSigma(-1.0)
	llama.SamplerChainAdd(sampler, topNSigma)

	topK := llama.SamplerInitTopK(40)
	llama.SamplerChainAdd(sampler, topK)

	typical := llama.SamplerInitTypical(1.0, 0)
	llama.SamplerChainAdd(sampler, typical)

	topP := llama.SamplerInitTopP(0.95, 0)
	llama.SamplerChainAdd(sampler, topP)

	minP := llama.SamplerInitMinP(0.05, 0)
	llama.SamplerChainAdd(sampler, minP)

	xtc := llama.SamplerInitXTC(0, 0.1, 0, llama.DEFAULT_SEED)
	llama.SamplerChainAdd(sampler, xtc)

	temp := llama.SamplerInitTempExt(0.2, 0, 1.0)
	llama.SamplerChainAdd(sampler, temp)

	// add this last
	dist := llama.SamplerInitDist(llama.DEFAULT_SEED)
	llama.SamplerChainAdd(sampler, dist)

	return sampler
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
