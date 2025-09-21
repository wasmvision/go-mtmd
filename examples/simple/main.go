package main

import (
	"fmt"

	"github.com/wasmvision/go-mtmd/pkg/llama"
	"github.com/wasmvision/go-mtmd/pkg/loader"
	"github.com/wasmvision/go-mtmd/pkg/mtmd"
)

var (
	file = "/home/ron/Development/gollama.cpp/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
)

func main() {
	fmt.Println("loading libs")
	lib := loader.LoadLibrary("./lib")

	fmt.Println("init libs")
	llama.Init(lib)
	mtmd.Init(lib)

	fmt.Println("loading all GGML backends")
	llama.GGMLBackendLoadAll()

	ctx := mtmd.ContextParamsDefault()
	fmt.Println("useGPU?", ctx.UseGPU)

	params := llama.LlamaModelDefaultParams()
	fmt.Println("MainGpu:", params.MainGpu)

	fmt.Println("Loading model", file)
	model := llama.LlamaModelLoadFromFile(file, params)

	fmt.Println("model free")
	llama.LlamaModelFree(model)

	fmt.Println("backend free")
	llama.BackendFree()
}
