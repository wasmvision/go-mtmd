package main

import (
	"fmt"

	"github.com/wasmvision/go-mtmd"
)

func main() {
	fmt.Println("loading libs")
	mtmd.LoadLibrary()

	fmt.Println("init libs")
	mtmd.Init()

	fmt.Println("backend init")
	mtmd.BackendInit()

	ctx := mtmd.ContextParamsDefault()
	fmt.Println("useGPU?", ctx.UseGPU)

	fmt.Println("backend free")
	mtmd.BackendFree()
}
