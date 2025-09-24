package main

import (
	"fmt"
	"image"
	"log"
	"os"
	"unsafe"

	"image/draw"
	_ "image/jpeg"

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
	fmt.Printf("%+v\n", params)

	fmt.Println("Loading model", modelFile)
	model := llama.ModelLoadFromFile(modelFile, params)
	defer func() {
		llama.ModelFree(model)
		fmt.Println("model free")
	}()

	//vocab := llama.ModelGetVocab(model)

	ctxParams := llama.ContextDefaultParams()
	fmt.Printf("%+v\n", ctxParams)

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

	tmpl := llama.ModelChatTemplate(model, "")
	fmt.Println("template", tmpl)

	marker := mtmd.DefaultMarker()
	fmt.Println("default marker", marker)

	mctxParams := mtmd.ContextParamsDefault()
	fmt.Printf("%+v\n", ctxParams)

	mctxParams.UseGPU = true
	mctxParams.Verbosity = llama.LogLevelError

	fmt.Println("Loading projector", projFile)
	mtmdCtx := mtmd.InitFromFile(projFile, model, mctxParams)
	defer func() {
		mtmd.Free(mtmdCtx)
		fmt.Println("mtmd context free")
	}()

	fmt.Println("Loading bitmap", imageFile)
	bitmap := mtmd.BitmapInitFromFile(mtmdCtx, imageFile) //openBitmapFromFile(imageFile)
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

	mtmd.HelperEvalChunks(mtmdCtx,
		lctx,                    // lctx
		output,                  // chunks
		0,                       // n_past
		0,                       // seq_id
		int32(ctxParams.NBatch), // n_batch
		true,                    // logits_last
		&n)

	fmt.Println("new pos", n)
}

func warmup(lctx llama.Context, model llama.Model, vocab llama.Vocab) {
	// llama_set_warmup(lctx, true);
	llama.SetWarmup(lctx, true)

	// std::vector<llama_token> tmp;
	tmp := make([]llama.Token, 0)

	// llama_token bos = llama_vocab_bos(vocab);
	bos := llama.VocabBOS(vocab)

	// llama_token eos = llama_vocab_eos(vocab);
	eos := llama.VocabEOS(vocab)

	// // some models (e.g. T5) don't have a BOS token
	// if (bos != LLAMA_TOKEN_NULL) {
	//     tmp.push_back(bos);
	// }
	if bos != llama.TOKEN_NULL {
		tmp = append(tmp, bos)
	}

	// if (eos != LLAMA_TOKEN_NULL) {
	//     tmp.push_back(eos);
	// }
	if eos != llama.TOKEN_NULL {
		tmp = append(tmp, eos)
	}

	// if (tmp.empty()) {
	//     tmp.push_back(0);
	// }
	if len(tmp) == 0 {
		tmp = append(tmp, 0)
	}

	// if (llama_model_has_encoder(model)) {
	if llama.ModelHasEncoder(model) {
		// llama_encode(lctx, llama_batch_get_one(tmp.data(), tmp.size()));
		batch := llama.BatchGetOne(unsafe.SliceData(tmp), int32(len(tmp)))
		llama.Encode(lctx, batch)

		// llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
		start := llama.ModelDecoderStartToken(model)
		// if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
		if start == llama.TOKEN_NULL {
			// decoder_start_token_id = bos;
			start = bos
		}
		// tmp.clear();
		// tmp.push_back(decoder_start_token_id);
		tmp = append([]llama.Token{}, start)
	}
	// if (llama_model_has_decoder(model)) {
	if llama.ModelHasDecoder(model) {
		//     llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) params.n_batch)));
		batch := llama.BatchGetOne(unsafe.SliceData(tmp), int32(len(tmp)))
		llama.Decode(lctx, batch)
	}

	// llama_memory_clear(llama_get_memory(lctx), true);
	mem := llama.GetMemory(lctx)
	llama.MemoryClear(mem, true)

	// llama_synchronize(lctx);
	llama.Synchronize(lctx)

	// llama_perf_context_reset(lctx);

	// llama_set_warmup(lctx, false);
	llama.SetWarmup(lctx, false)
}

func openBitmapFromFile(filename string) mtmd.Bitmap {
	fmt.Println("Opening image", filename)

	reader, err := os.Open(filename)
	if err != nil {
		panic(err)
	}

	src, _, err := image.Decode(reader)
	if err != nil {
		log.Fatal(err)
	}

	b := src.Bounds()
	img := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(img, img.Bounds(), src, b.Min, draw.Src)

	data := make([]byte, b.Dx()*b.Dy()*3)
	i := 0
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			px := img.At(x, y)

			r, g, b, _ := px.RGBA()
			data[i], data[i+1], data[i+2] = byte(r), byte(g), byte(b)
			i += 3
		}
	}

	fmt.Println("bitmap init")
	bitmap := mtmd.BitmapInit(uint32(b.Dx()), uint32(b.Dy()), uintptr(unsafe.Pointer(unsafe.SliceData(data))))

	return bitmap
}
