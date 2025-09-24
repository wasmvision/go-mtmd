package main

import (
	"fmt"
	"unsafe"

	"github.com/wasmvision/go-mtmd/pkg/llama"
	"github.com/wasmvision/go-mtmd/pkg/loader"
	"github.com/wasmvision/go-mtmd/pkg/mtmd"
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

	// fmt.Println("warming up")
	// warmup(lctx, model, vocab)

	mctxParams := mtmd.ContextParamsDefault()
	fmt.Printf("mtmd Context Params: %+v\n", mctxParams)

	// mctxParams.UseGPU = true
	// mctxParams.Verbosity = llama.LogLevelError

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
	mtmd.Tokenize(mtmdCtx, output, input, bt, 1)

	var n llama.Pos
	mtmd.HelperEvalChunks(mtmdCtx, lctx, output, 0, 0, int32(ctxParams.NBatch), true, &n)

	fmt.Println("new pos", n)

	sampler := setupSampler()

	// batch = llama_batch_init(1, 0, 1); // batch for next token generation
	batch := llama.BatchInit(1, 0, 1)
	// for (int i = 0; i < n_predict; i++) {
	//tokens := []llama.Token{}

	for {
		//     if (i > n_predict || !g_is_generating || g_is_interrupted) {
		//         LOG("\n");
		//         break;
		//     }

		//     llama_token token_id = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
		token := llama.SamplerSample(sampler, lctx, -1)
		//     generated_tokens.push_back(token_id);
		//tokens = append(tokens, token)
		//     common_sampler_accept(ctx.smpl, token_id, true);
		llama.SamplerAccept(sampler, token)

		//     if (llama_vocab_is_eog(ctx.vocab, token_id) || ctx.check_antiprompt(generated_tokens)) {
		if llama.VocabIsEOG(vocab, token) {
			// end of generation
			fmt.Println()
			break
		}

		//     LOG("%s", common_token_to_piece(ctx.lctx, token_id).c_str());
		//     fflush(stdout);

		data := make([]byte, 24)
		buf := unsafe.SliceData(data)
		llama.TokenToPiece(vocab, token, buf, 24, 0, true)

		fmt.Print(string(data))
		//     if (g_is_interrupted) {
		//         LOG("\n");
		//         break;
		//     }

		//     // eval the token
		//     common_batch_clear(ctx.batch);

		//     common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
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

func setupSampler() llama.Sampler {
	// sparams = llama_sampler_chain_default_params();
	params := llama.SamplerChainDefaultParams()

	// llama_sampler * smpl = llama_sampler_chain_init(sparams);
	sampler := llama.SamplerChainInit(params)

	// llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
	greedy := llama.SamplerInitGreedy()
	llama.SamplerChainAdd(sampler, greedy)

	return sampler
}
