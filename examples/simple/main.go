package main

import (
	"fmt"

	"github.com/wasmvision/go-mtmd"
)

var (
	file = "/home/ron/Development/gollama.cpp/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
)

func main() {
	fmt.Println("loading libs")
	mtmd.LoadLibrary()

	fmt.Println("init libs")
	mtmd.Init()

	// fmt.Println("backend init")
	// mtmd.BackendInit()

	fmt.Println("loading all GGML backends")
	mtmd.GGMLBackendLoadAll()

	ctx := mtmd.ContextParamsDefault()
	fmt.Println("useGPU?", ctx.UseGPU)

	params := mtmd.LlamaModelDefaultParams()
	fmt.Println("MainGpu:", params.MainGpu)

	fmt.Println("Loading model", file)
	model := mtmd.LlamaModelLoadFromFile(file, params)

	fmt.Println("model free")
	mtmd.LlamaModelFree(model)

	fmt.Println("backend free")
	mtmd.BackendFree()
}
