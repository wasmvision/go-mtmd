package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"os"
	"unsafe"

	"github.com/wasmvision/yzma/pkg/llama"
	"github.com/wasmvision/yzma/pkg/loader"
	"golang.org/x/sys/unix"
)

var (
	modelFile *string
	prompt    *string
	template  *string
	maxTokens *int
	libPath   *string
	verbose   *bool

	vocab   llama.Vocab
	model   llama.Model
	lctx    llama.Context
	sampler llama.Sampler

	messages []llama.ChatMessage
)

func main() {
	if err := handleFlags(); err != nil {
		showUsage()
		os.Exit(0)
	}

	lib, err := loader.LoadLibrary(*libPath)
	if err != nil {
		fmt.Println("unable to load library", err.Error())
		os.Exit(1)
	}
	if err := llama.Load(lib); err != nil {
		fmt.Println("unable to load library", err.Error())
		os.Exit(1)
	}

	llama.BackendInit()
	defer llama.BackendFree()

	if !*verbose {
		llama.LogSet(llama.LogSilent(), uintptr(0))
	}

	model = llama.ModelLoadFromFile(*modelFile, llama.ModelDefaultParams())
	defer llama.ModelFree(model)

	vocab = llama.ModelGetVocab(model)
	sampler = llama.SamplerChainInit(llama.SamplerChainDefaultParams())
	llama.SamplerChainAdd(sampler, llama.SamplerInitGreedy())

	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = 4096
	ctxParams.NBatch = 2048

	lctx = llama.InitFromModel(model, ctxParams)
	defer llama.Free(lctx)

	messages = make([]llama.ChatMessage, 0)

	// single message
	if len(*prompt) > 0 {
		messages = append(messages, llama.NewChatMessage("user", *prompt))
		chat(chatTemplate(true))

		return
	}

	// chat session
	first := true
	for {
		fmt.Print("Enter prompt: ")
		reader := bufio.NewReader(os.Stdin)
		pmpt, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println("unable to read user input", err.Error())
			os.Exit(1)
		}

		messages = append(messages, llama.NewChatMessage("user", pmpt))
		chat(chatTemplate(first))
		first = false
	}
}

func chat(text string) {
	// call once to get the size
	count := llama.Tokenize(vocab, text, nil, true, true)

	// now get the actual tokens
	tokens := make([]llama.Token, count)
	llama.Tokenize(vocab, text, tokens, true, true)

	batch := llama.BatchGetOne(tokens)

	if llama.ModelHasEncoder(model) {
		llama.Encode(lctx, batch)

		start := llama.ModelDecoderStartToken(model)
		if start == llama.TOKEN_NULL {
			start = llama.VocabBOS(vocab)
		}

		batch = llama.BatchGetOne([]llama.Token{start})
	}

	fmt.Println()

	response := ""
	for pos := int32(0); pos+batch.NTokens < int32(*maxTokens); pos += batch.NTokens {
		llama.Decode(lctx, batch)

		token := llama.SamplerSample(sampler, lctx, -1)
		llama.SamplerAccept(sampler, token)

		if llama.VocabIsEOG(vocab, token) {
			messages = append(messages, llama.NewChatMessage("assistant", response))
			fmt.Println()
			break
		}

		buf := make([]byte, 16)
		l := llama.TokenToPiece(vocab, token, buf, 0, true)
		next := string(buf[:l])

		batch = llama.BatchGetOne([]llama.Token{token})

		// TODO: figure out why qwen returns this.
		if next == "assistant" {
			continue
		}

		fmt.Print(next)
		response += next
	}

	fmt.Println()
}

func chatTemplate(first bool) string {
	buf := make([]byte, 1024)
	len := llama.ChatApplyTemplate(*template, messages, first, buf)
	return unix.BytePtrToString(unsafe.SliceData(buf[:len]))
}

func showUsage() {
	fmt.Println(`
Usage:
chat -model [model file path] -lib [llama.cpp .so file path] -prompt [omit this flag for a chat session] -v`)
}

func handleFlags() error {
	modelFile = flag.String("model", "", "model file to use")
	prompt = flag.String("prompt", "", "prompt")
	template = flag.String("template", "chatml", "template name (defaults to chatml)")
	maxTokens = flag.Int("maxtokens", -1, "maximum number of tokens to process")
	libPath = flag.String("lib", "", "path to llama.cpp compiled library files")
	verbose = flag.Bool("v", false, "verbose logging")

	flag.Parse()

	if len(*modelFile) == 0 ||
		len(*libPath) == 0 {

		return errors.New("missing a flag")
	}

	if *maxTokens < 0 {
		*maxTokens = llama.MaxToken
	}

	return nil
}
