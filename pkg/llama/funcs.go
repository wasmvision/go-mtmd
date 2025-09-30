package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
)

var (
	backendInitFunc ffi.Fun

	backendFreeFunc ffi.Fun

	// GGML_API void ggml_backend_load_all(void);
	ggmlBackendLoadAllFunc ffi.Fun

	// GGML_API void ggml_backend_load_all(void);
	ggmlBackendLoadAllFromPath ffi.Fun

	// LLAMA_API void llama_memory_clear(
	// 	llama_memory_t mem,
	// 				bool data);
	memoryClearFunc ffi.Fun

	// LLAMA_API           llama_memory_t   llama_get_memory  (const struct llama_context * ctx);
	getMemoryFunc ffi.Fun

	// LLAMA_API void llama_synchronize(struct llama_context * ctx);
	synchronizeFunc ffi.Fun
)

func loadFuncs(lib ffi.Lib) error {
	var err error
	if backendInitFunc, err = lib.Prep("llama_backend_init", &ffi.TypeVoid); err != nil {
		return err
	}

	if backendFreeFunc, err = lib.Prep("llama_backend_free", &ffi.TypeVoid); err != nil {
		return err
	}

	if ggmlBackendLoadAllFunc, err = lib.Prep("ggml_backend_load_all", &ffi.TypeVoid); err != nil {
		return err
	}

	if ggmlBackendLoadAllFromPath, err = lib.Prep("ggml_backend_load_all_from_path", &ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return err
	}

	if memoryClearFunc, err = lib.Prep("llama_memory_clear", &ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeUint8); err != nil {
		return err
	}

	if getMemoryFunc, err = lib.Prep("llama_get_memory", &ffi.TypePointer, &ffi.TypePointer); err != nil {
		return err
	}

	if synchronizeFunc, err = lib.Prep("llama_synchronize", &ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return err
	}

	return nil
}

// BackendInit initializes the llama.cpp back-end.
func BackendInit() {
	backendInitFunc.Call(nil)
}

// BackendFree frees the llama.cpp back-end.
func BackendFree() {
	backendFreeFunc.Call(nil)
}

// GGMLBackendLoadAll loads all backends using the default search paths.
func GGMLBackendLoadAll() {
	ggmlBackendLoadAllFunc.Call(nil)
}

// GGMLBackendLoadAllFromPath loads all backends from a specific path.
func GGMLBackendLoadAllFromPath(path string) {
	p := &[]byte(path + "\x00")[0]
	ggmlBackendLoadAllFromPath.Call(nil, unsafe.Pointer(&p))
}

func MemoryClear(mem Memory, data bool) {
	memoryClearFunc.Call(nil, unsafe.Pointer(&mem), unsafe.Pointer(&data))
}

func GetMemory(ctx Context) Memory {
	var mem Memory
	getMemoryFunc.Call(unsafe.Pointer(&mem), unsafe.Pointer(&ctx))

	return mem
}

func Synchronize(ctx Context) {
	synchronizeFunc.Call(nil, unsafe.Pointer(&ctx))
}
