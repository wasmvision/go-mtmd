package main

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/wasmvision/go-mtmd/pkg/llama"
	"github.com/wasmvision/go-mtmd/pkg/loader"
	"github.com/wasmvision/go-mtmd/pkg/mtmd"
	"golang.org/x/sys/unix"
)

var (
	modelFile = "/home/ron/Development/go-mtmd/models/Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
	projFile  = "/home/ron/Development/go-mtmd/models/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
	imageFile = "/home/ron/Development/go-mtmd/images/roneye_400x400.jpg"

	prompt = "what is this?"
)

func chatTemplate(prompt string) string {
	return fmt.Sprintf(`<|im_start|>user                                                                                                                                                                             
%s<__media__><|im_end|>
<|im_start|>assistant`, prompt)
}

func main() {
	fmt.Println("loading libs")
	lib := loader.LoadLibrary(".")

	fmt.Println("init libs")
	llama.Init(lib)
	mtmd.Init(lib)

	fmt.Println("loading all GGML backends")
	llama.GGMLBackendLoadAll()
	defer func() {
		fmt.Println("backend free")
		llama.BackendFree()
	}()

	params := llama.ModelDefaultParams()
	fmt.Printf("Model params: %+v\n", params)

	fmt.Println("Loading model", modelFile)
	model := llama.ModelLoadFromFile(modelFile, params)
	defer func() {
		llama.ModelFree(model)
		fmt.Println("model free")
	}()

	vocab := llama.ModelGetVocab(model)

	ctxParams := llama.ContextDefaultParams()
	fmt.Printf("Context params: %+v\n", ctxParams)

	ctxParams.NCtx = 4096
	ctxParams.NBatch = 2048

	fmt.Println("Init model")
	lctx := llama.InitFromModel(model, ctxParams)
	defer func() {
		llama.Free(lctx)
		fmt.Println("llama context free")
	}()

	sampler := setupSampler(vocab)

	// fmt.Println("warming up")
	// utils.Warmup(lctx, model, vocab)

	mctxParams := mtmd.ContextParamsDefault()
	fmt.Printf("mtmd Context Params: %+v\n", mctxParams)

	// mctxParams.UseGPU = true
	mctxParams.Verbosity = llama.LogLevelError

	fmt.Println("Loading projector", projFile)
	mtmdCtx := mtmd.InitFromFile(projFile, model, mctxParams)
	defer func() {
		mtmd.Free(mtmdCtx)
		fmt.Println("mtmd context free")
	}()

	fmt.Println("Loading bitmap", imageFile)
	bitmap := mtmd.BitmapInitFromFile(mtmdCtx, imageFile)
	defer func() {
		mtmd.BitmapFree(bitmap)
		fmt.Println("bitmap free")
	}()

	fmt.Println("bitmap size", mtmd.BitmapGetNBytes(bitmap))

	fmt.Println("loading template")
	pmpt := chatTemplate(prompt)
	input := &mtmd.InputText{
		Text:         &[]byte(pmpt + "\x00")[0],
		AddSpecial:   true,
		ParseSpecial: true,
	}

	output := mtmd.InputChunksInit()

	bitmaps := []mtmd.Bitmap{bitmap}
	bt := unsafe.SliceData(bitmaps)

	fmt.Println("tokenize")
	stats := mtmd.Tokenize(mtmdCtx, output, input, bt, 1)
	fmt.Println("tokenize result", stats)

	fmt.Println("got chunks", mtmd.InputChunksSize(output))

	var n llama.Pos
	res := mtmd.HelperEvalChunks(mtmdCtx, lctx, output, 0, 0, int32(ctxParams.NBatch), true, &n)

	fmt.Println("res", res, "new pos", n)

	batch := llama.BatchInit(1, 0, 1)
	for {
		token := llama.SamplerSample(sampler, lctx, -1)
		llama.SamplerAccept(sampler, token)

		if llama.VocabIsEOG(vocab, token) {
			// end of generation
			fmt.Println()
			fmt.Println("Done.")
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

func setupSampler(vocab llama.Vocab) llama.Sampler {
	params := llama.SamplerChainDefaultParams()
	sampler := llama.SamplerChainInit(params)

	// for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
	//     if (llama_vocab_is_eog(vocab, i)) {
	//         LOG_INF("%s: added %s logit bias = %f\n", __func__, common_token_to_piece(lctx, i).c_str(), -INFINITY);
	//         params.sampling.logit_bias_eog.push_back({i, -INFINITY});
	//     }
	// }

	// if (params.sampling.ignore_eos) {
	//     // add EOG biases to the active set of logit biases
	//     params.sampling.logit_bias.insert(
	//             params.sampling.logit_bias.end(),
	//             params.sampling.logit_bias_eog.begin(), params.sampling.logit_bias_eog.end());
	// }
	logitBiasEOG := make([]llama.LogitBias, 0)
	nTokens := llama.VocabNTokens(vocab)

	for i := int32(0); i < nTokens; i++ {
		token := llama.Token(i)
		if llama.VocabIsEOG(vocab, token) {
			logitBiasEOG = append(logitBiasEOG, llama.LogitBias{Token: token, Bias: -1 * math.MaxFloat32})
		}
	}

	bias := llama.SamplerInitLogitBias(nTokens, int32(len(logitBiasEOG)), unsafe.SliceData(logitBiasEOG))
	llama.SamplerChainAdd(sampler, bias)

	// greedy := llama.SamplerInitGreedy()
	// llama.SamplerChainAdd(sampler, greedy)
	dist := llama.SamplerInitDist(llama.DEFAULT_SEED)
	llama.SamplerChainAdd(sampler, dist)

	return sampler
}
