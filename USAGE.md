# Usage

Here are a few examples of yzma with language models that appear to work.

## VLM using Qwen2.5-VL-3B-Instruct-Q8_0

```
go run ./examples/vlm/ -model ~/models/Qwen2.5-VL-3B-Instruct-Q8_0.gguf -proj ~/models/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf -image ./images/domestic_llama.jpg -prompt "What is in this picture?" 2>/dev/null
```

## VLM using moondream2-20250414-GGUF

```
go run ./examples/vlm/ -model ~/models/moondream2-text-model-f16_ct-vicuna.gguf -proj ~/models/moondream2-mmproj-f16-20250414.gguf -image ./images/domestic_llama.jpg -prompt "What is in this picture?" 2>/dev/null
```

## Chat using qwen2.5-0.5b-instruct-fp16

```
go run ./demo/chat/ -model ~/models/qwen2.5-0.5b-instruct-fp16.gguf -temp=0.6 -n=512
```

## Chat using tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

```
go run ./demo/chat/ -model ~/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -c 2048 -temp 0.7 -n 512
```

## Chat using gemma-3-1b-it-Q4_K_M.gguf

https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/blob/main/gemma-3-1b-it-Q4_K_M.gguf

```
go run ./examples/chat/ -model ~/models/gemma-3-1b-it-Q4_K_M.gguf
```
