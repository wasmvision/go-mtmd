# yzma

Go package that lets you call llama.cpp to perform multimodal inference using Vision Language Models (VLMs).

Uses `purego` and `ffi` packages so CGo is not required.

Still a work in progress, but is minimally functioning.

Borrows definitions from the https://github.com/dianlight/gollama.cpp package then modifies them rather heavily. Thank you!

***Important Note***
You have to add the directory with your llama.cpp .so files to your `LD_LIBRARY_PATH` env variable:

```shell
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/ron/Development/yzma/lib
```

## Examples

### VLM example

```shell
$ go run ./examples/vlm/ -model ./models/Qwen2.5-VL-3B-Instruct-Q8_0.gguf -proj ./models/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf -lib ./lib -image ./images/domestic_llama.jpg -prompt "What is in this picture?" 2>/dev/null
Loading model ./models/Qwen2.5-VL-3B-Instruct-Q8_0.gguf
encoding image slice...
image slice encoded in 966 ms
decoding image batch 1/1, n_tokens_batch = 910
image decoded (batch 1/1) in 208 ms

The picture shows a white llama standing in a fenced-in area, possibly a zoo or a wildlife park. The llama is the main focus of the image, and it appears to be looking to the right. The background features a grassy area with trees and a fence, and there are some vehicles visible in the distance.
```

### Chat example

```shell
$ go run ./examples/chat/ -model ./models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf -lib ./lib -prompt "Are you ready to go?"                                                   
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no                                                                                                                               
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no                                                                                                                               
ggml_cuda_init: found 1 CUDA devices:                                               
  Device 0: NVIDIA GeForce RTX 4070 Laptop GPU, compute capability 8.9, VMM: yes    
register_backend: registered backend CUDA (1 devices)                               
register_device: registered device CUDA0 (NVIDIA GeForce RTX 4070 Laptop GPU)       
register_backend: registered backend CPU (1 devices)                                
register_device: registered device CPU (13th Gen Intel(R) Core(TM) i9-13900HX)                                                                                           
llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GeForce RTX 4070 Laptop GPU) (0000:01:00.0) - 7657 MiB free                                                  
llama_model_loader: loaded meta data with 23 key-value pairs and 201 tensors from ./models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf (version GGUF V3 (latest))
...
llama_kv_cache:      CUDA0 KV buffer size =     5.50 MiB                                                                                                                 
llama_kv_cache: size =    5.50 MiB (   256 cells,  22 layers,  1/1 seqs), K (f16):    2.75 MiB, V (f16):    2.75 MiB                                                     
llama_context: enumerating backends                                                                                                                                      
llama_context: backend_ptrs.size() = 2                                              
llama_context: max_nodes = 1608                                                     
llama_context: reserving full memory module                                         
llama_context: worst-case: n_tokens = 64, n_seqs = 1, n_outputs = 1                                                                                                      
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1                                                                          
llama_context: Flash Attention was auto, set to enabled                                                                                                                  
graph_reserve: reserving a graph for ubatch with n_tokens =   64, n_seqs =  1, n_outputs =   64                                                                          
ggml_gallocr_reserve_n: reallocating CUDA0 buffer from size 0.00 MiB to 8.31 MiB    
ggml_gallocr_reserve_n: reallocating CUDA_Host buffer from size 0.00 MiB to 0.56 MiB                                                                                     
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1                                                                          
graph_reserve: reserving a graph for ubatch with n_tokens =   64, n_seqs =  1, n_outputs =   64                                                                          
llama_context:      CUDA0 compute buffer size =     8.31 MiB                        
llama_context:  CUDA_Host compute buffer size =     0.56 MiB                        
llama_context: graph nodes  = 689                                                   
llama_context: graph splits = 2                                                     
                                                                                    
check_node_graph_compatibility_and_refresh_copy_ops: disabling CUDA graphs due to batch size > 1 [ffn_inp-0] [2048 7 1 1]                                                
                                                                                                                                                                         
                                                                                                                                                                         
JASON: (smiling) Yeah, I'm ready to go.
```
