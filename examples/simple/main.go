package main

import (
	"fmt"

	"github.com/wasmvision/yzma/pkg/llama"
	"github.com/wasmvision/yzma/pkg/loader"
)

var (
	modelFile = "/home/ron/Development/yzma/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
)

func main() {
	fmt.Println("loading libs")
	lib := loader.LoadLibrary(".")

	fmt.Println("init libs")
	llama.Init(lib)

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
}
