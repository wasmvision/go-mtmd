# yzma

`yzma` lets you perform multimodal inference with Vision Language Models (VLMs) on your own hardware by using the [`llama.cpp`](https://github.com/ggml-org/llama.cpp) libraries.

It uses the [`purego`](https://github.com/ebitengine/purego) and [`ffi`](https://github.com/JupiterRider/ffi) packages so calls can be made directly to `llama.cpp` without CGo.

Still a work in progress but it is already functioning.

Borrows definitions from the https://github.com/dianlight/gollama.cpp package then modifies them rather heavily. Thank you!

## Installation

You will need to download the `llama.cpp` libraries for your platform. You can obtain them from https://github.com/ggml-org/llama.cpp/releases

Extract the library files into a directory on local machine.

For Linux, they have the `.so` file extension. For example, `libllama.so`, `libmtmd.so` and so on. When using macOS, they have a `.dylib` file extension. And on Windows, they have a `.dll` file extension. You do not need the other downloaded files to use the `llama.cpp` libraries with `yzma`.

***Important Note***
You will need to add the directory with your llama.cpp library files to your `LD_LIBRARY_PATH` env variable. For example:

```shell
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/ron/Development/yzma/lib
```

## Examples

### VLM example

This example uses the [`Qwen2.5-VL-3B-Instruct-Q8_0`](https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF) VLM model to process a input of a text prompt and an image, and then displays the result.

```shell
$ go run ./examples/vlm/ -model ./models/Qwen2.5-VL-3B-Instruct-Q8_0.gguf -proj ./models/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf -lib ./lib -image ./images/domestic_llama.jpg -prompt "What is in this picture?" 2>/dev/null
Loading model ./models/Qwen2.5-VL-3B-Instruct-Q8_0.gguf
encoding image slice...
image slice encoded in 966 ms
decoding image batch 1/1, n_tokens_batch = 910
image decoded (batch 1/1) in 208 ms

The picture shows a white llama standing in a fenced-in area, possibly a zoo or a wildlife park. The llama is the main focus of the image, and it appears to be looking to the right. The background features a grassy area with trees and a fence, and there are some vehicles visible in the distance.
```

[See the code here](./examples/vlm/main.go).

### Chat example

You can also use `yzma` to do inference on text language models. This example uses the [`qwen2.5-0.5b-instruct-fp16.gguf `](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) model for a chat session.


```shell
$ go run ./examples/chat/ -model ./models/qwen2.5-0.5b-instruct-fp16.gguf -lib ./lib/
Enter prompt: Are you ready to go?

Yes, I'm ready to go! What would you like to do?

Enter prompt: Let's go to the zoo


Great! Let's go to the zoo. What would you like to see?

Enter prompt: I want to feed the llama 


Sure! Let's go to the zoo and feed the llama. What kind of llama are you interested in feeding?
```

[See the code here](./examples/chat/main.go).
