# go-mtmd


```shell
$ go run ../examples/simple/                                                                              
loading libs                                                                                              
init libs                                                                                                 
loading all GGML backends                                                                                 
load_backend: loaded RPC backend from /home/ron/Development/go-mtmd/lib/libggml-rpc.so                    
load_backend: loaded CPU backend from /home/ron/Development/go-mtmd/lib/libggml-cpu-alderlake.so          
useGPU? true                                                                                              
MainGpu: 0                                                                                                
Loading model /home/ron/Development/gollama.cpp/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf                                                                                                                           
llama_model_loader: loaded meta data with 23 key-value pairs and 201 tensors from /home/ron/Development/gollama.cpp/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.         
llama_model_loader: - kv   0:                       general.architecture str              = llama         
llama_model_loader: - kv   1:                               general.name str              = tinyllama_tinyllama-1.1b-chat-v1.0
llama_model_loader: - kv   2:                       llama.context_length u32              = 2048          
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 2048
llama_model_loader: - kv   4:                          llama.block_count u32              = 22            
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 5632          
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 64            
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32            
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 4
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                       llama.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  11:                          general.file_type u32              = 10
llama_model_loader: - kv  12:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  13:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  14:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  15:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  16:                      tokenizer.ggml.merges arr[str,61249]   = ["▁ t", "e r", "i n", "▁ a", "e n...
llama_model_loader: - kv  17:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  18:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  19:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  20:            tokenizer.ggml.padding_token_id u32              = 2
llama_model_loader: - kv  21:                    tokenizer.chat_template str              = {% for message in messages %}\n{% if m...
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   45 tensors
llama_model_loader: - type q2_K:   45 tensors
llama_model_loader: - type q3_K:  110 tensors
llama_model_loader: - type q6_K:    1 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q2_K - Medium
print_info: file size   = 459.11 MiB (3.50 BPW)
```
