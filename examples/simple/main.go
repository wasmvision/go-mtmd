package main

import (
	"fmt"

	"github.com/wasmvision/go-mtmd/pkg/llama"
	"github.com/wasmvision/go-mtmd/pkg/loader"
	"github.com/wasmvision/go-mtmd/pkg/mtmd"
)

var (
	//file = "/home/ron/Development/gollama.cpp/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
	modelFile = "/home/ron/Development/go-mtmd/models/Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
	projFile  = "/home/ron/Development/go-mtmd/models/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
)

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

	ctxParams := mtmd.ContextParamsDefault()
	fmt.Printf("%+v\n", ctxParams)

	ctxParams.UseGPU = false
	ctxParams.Verbosity = llama.LogLevelError

	fmt.Println("Loading projector", projFile)
	mtmdCtx := mtmd.InitFromFile(projFile, model, ctxParams)
	defer func() {
		mtmd.Free(mtmdCtx)
		fmt.Println("mtmd context free")
	}()
}
