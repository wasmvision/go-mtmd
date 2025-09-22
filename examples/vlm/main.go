package main

import (
	"fmt"
	"image"
	"log"
	"os"
	"unsafe"

	_ "image/jpeg"

	"github.com/wasmvision/go-mtmd/pkg/llama"
	"github.com/wasmvision/go-mtmd/pkg/loader"
	"github.com/wasmvision/go-mtmd/pkg/mtmd"
)

var (
	modelFile = "/home/ron/Development/go-mtmd/models/Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
	projFile  = "/home/ron/Development/go-mtmd/models/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
	imageFile = "/home/ron/Development/go-mtmd/images/roneye_400x400.jpg"
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

	bitmap := openBitmapFromFile(imageFile)
	defer func() {
		mtmd.BitmapFree(bitmap)
		fmt.Println("bitmap free")
	}()
}

func openBitmapFromFile(filename string) mtmd.Bitmap {
	fmt.Println("Opening image", filename)

	reader, err := os.Open(filename)
	if err != nil {
		panic(err)
	}

	img, _, err := image.Decode(reader)
	if err != nil {
		log.Fatal(err)
	}

	sz := img.Bounds().Size()
	data := make([]uint32, sz.X*sz.Y*3)
	i := 0
	for y := 0; y < sz.Y; y++ {
		for x := 0; x < sz.X; x++ {
			px := img.At(x, y)

			data[i], data[i+1], data[i+2], _ = px.RGBA()
			i += 3
		}
	}

	fmt.Println("bitmap init")
	bitmap := mtmd.BitmapInit(uint32(sz.X), uint32(sz.Y), uintptr(unsafe.Pointer(unsafe.SliceData(data))))

	return bitmap
}
