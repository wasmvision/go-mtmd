package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
)

var (
	// LLAMA_API const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
	ModelGetVocab     func(model Model) Vocab
	modelGetVocabFunc ffi.Fun

	// LLAMA_API llama_token llama_vocab_bos(const struct llama_vocab * vocab); // beginning-of-sentence
	VocabBOS     func(vocab Vocab) Token
	vocabBOSFunc ffi.Fun

	// LLAMA_API llama_token llama_vocab_eos(const struct llama_vocab * vocab); // end-of-sentence
	VocabEOS     func(vocab Vocab) Token
	vocabEOSFunc ffi.Fun

	// LLAMA_API bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);
	VocabIsEOG     func(vocab Vocab, token Token) bool
	vocabIsEOGFunc ffi.Fun
)

func initVocab(lib ffi.Lib) {
	var err error
	modelGetVocabFunc, err = lib.Prep("llama_model_get_vocab", &ffi.TypePointer, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	ModelGetVocab = func(model Model) Vocab {
		var vocab Vocab
		modelGetVocabFunc.Call(unsafe.Pointer(&vocab), unsafe.Pointer(&model))

		return vocab
	}

	vocabBOSFunc, err = lib.Prep("llama_vocab_bos", &ffi.TypeSint32, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	VocabBOS = func(vocab Vocab) Token {
		var token ffi.Arg
		vocabBOSFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))

		return Token(token)
	}

	vocabEOSFunc, err = lib.Prep("llama_vocab_eos", &ffi.TypeSint32, &ffi.TypePointer)
	if err != nil {
		panic(err)
	}

	VocabEOS = func(vocab Vocab) Token {
		var token ffi.Arg
		vocabEOSFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))

		return Token(token)
	}

	vocabIsEOGFunc, err = lib.Prep("llama_vocab_is_eog", &ffi.TypeUint8, &ffi.TypePointer, &ffi.TypeSint32)
	if err != nil {
		panic(err)
	}

	VocabIsEOG = func(vocab Vocab, token Token) bool {
		var result ffi.Arg
		vocabIsEOGFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&vocab), unsafe.Pointer(&token))

		return result.Bool()
	}
}
