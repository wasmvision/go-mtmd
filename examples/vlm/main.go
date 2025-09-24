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
