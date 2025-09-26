# yzma

Go package that lets you call llama.cpp to perform multimodal inference using Vision Language Models (VLMs).

Uses `purego` and `ffi` packages so CGo is not required.

Still a work in progress, but is minimally functioning.

Borrows definitions from the https://github.com/dianlight/gollama.cpp package then modifies them rather heavily. Thank you!

```shell
$ go run ../examples/vlm/                                                                              
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070 Laptop GPU, compute capability 8.9, VMM: yes
register_backend: registered backend CUDA (1 devices)
register_device: registered device CUDA0 (NVIDIA GeForce RTX 4070 Laptop GPU)
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (13th Gen Intel(R) Core(TM) i9-13900HX)
load_backend: failed to find ggml_backend_init in /home/ron/Development/yzma/lib/libggml-cuda.so
load_backend: loaded RPC backend from /home/ron/Development/yzma/lib/libggml-rpc.so
register_backend: registered backend RPC (0 devices)
ggml_backend_load_best: /home/ron/Development/yzma/lib/libggml-cpu-sse42.so score: 5
ggml_backend_load_best: /home/ron/Development/yzma/lib/libggml-cpu-haswell.so score: 64
ggml_backend_load_best: /home/ron/Development/yzma/lib/libggml-cpu-sapphirerapids.so score: 0
ggml_backend_load_best: /home/ron/Development/yzma/lib/libggml-cpu-alderlake.so score: 128
ggml_backend_load_best: /home/ron/Development/yzma/lib/libggml-cpu-sandybridge.so score: 21
ggml_backend_load_best: /home/ron/Development/yzma/lib/libggml-cpu-x64.so score: 1
ggml_backend_load_best: /home/ron/Development/yzma/lib/libggml-cpu-skylakex.so score: 0
ggml_backend_load_best: /home/ron/Development/yzma/lib/libggml-cpu-icelake.so score: 0
load_backend: loaded CPU backend from /home/ron/Development/yzma/lib/libggml-cpu-alderlake.so
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (13th Gen Intel(R) Core(TM) i9-13900HX)
Loading model /home/ron/Development/yzma/models/Qwen2.5-VL-3B-Instruct-Q8_0.gguf
llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GeForce RTX 4070 Laptop GPU) (0000:01:00.0) - 7657 MiB free
llama_model_loader: loaded meta data with 27 key-value pairs and 434 tensors from /home/ron/Development/yzma/models/Qwen2.5-VL-3B-Instruct-Q8_0.gguf (version GGUF V3 (latest))
...
--- vision hparams ---
load_hparams: image_size:         1024
load_hparams: patch_size:         14
load_hparams: has_llava_proj:     0
load_hparams: minicpmv_version:   0
load_hparams: proj_scale_factor:  0
load_hparams: n_wa_pattern:       8

load_hparams: model size:         805.59 MiB
load_hparams: metadata size:      0.18 MiB
ggml_gallocr_reserve_n: reallocating CUDA0 buffer from size 0.00 MiB to 3.60 MiB
ggml_gallocr_reserve_n: reallocating CPU buffer from size 0.00 MiB to 0.16 MiB
alloc_compute_meta:      CUDA0 compute buffer size =     3.60 MiB
alloc_compute_meta:        CPU compute buffer size =     0.16 MiB
check_node_graph_compatibility_and_refresh_copy_ops: disabling CUDA graphs due to batch size > 1 [ffn_inp-0] [2048 8 1 1]
ggml_gallocr_needs_realloc: src 0 (inp_raw) of node inp_raw (view) is not valid
ggml_gallocr_alloc_graph: cannot reallocate multi buffer graph automatically, call reserve
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
ggml_gallocr_reserve_n: reallocating CUDA0 buffer from size 3.60 MiB to 76.54 MiB
ggml_gallocr_reserve_n: reallocating CPU buffer from size 0.16 MiB to 5.12 MiB
check_node_graph_compatibility_and_refresh_copy_ops: disabling CUDA graphs due to batch size > 1 [node_14] [30 30 1280 1]
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
check_node_graph_compatibility_and_refresh_copy_ops: disabling CUDA graphs due to batch size > 1 [ffn_inp-0] [2048 225 1 1]
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
check_node_graph_compatibility_and_refresh_copy_ops: disabling CUDA graphs due to batch size > 1 [ffn_inp-0] [2048 6 1 1]

This is a close-up image of a person's eye.
```
