package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
	"golang.org/x/sys/unix"
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

	// LLAMA_API int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);
	VocabNTokens     func(vocab Vocab) int32
	vocabNTokensFunc ffi.Fun

	// LLAMA_API int32_t llama_token_to_piece(
	// 		const struct llama_vocab * vocab,
	// 					llama_token   token,
	// 							char * buf,
	// 						int32_t   length,
	// 						int32_t   lstrip,
	// 							bool   special);
	TokenToPiece     func(vocab Vocab, token Token, buf []byte, lstrip int32, special bool) int32
	tokenToPieceFunc ffi.Fun

	// LLAMA_API int32_t llama_tokenize(
	//     const struct llama_vocab * vocab,
	//                   const char * text,
	//                      int32_t   text_len,
	//                  llama_token * tokens,
	//                      int32_t   n_tokens_max,
	//                         bool   add_special,
	//                         bool   parse_special);
	Tokenize     func(vocab Vocab, text string, tokens []Token, addSpecial bool, parseSpecial bool) int32
	tokenizeFunc ffi.Fun
)

func loadVocabFuncs(lib ffi.Lib) {
	var err error
	if modelGetVocabFunc, err = lib.Prep("llama_model_get_vocab", &ffi.TypePointer, &ffi.TypePointer); err != nil {
		panic(err)
	}
	ModelGetVocab = func(model Model) Vocab {
		var vocab Vocab
		modelGetVocabFunc.Call(unsafe.Pointer(&vocab), unsafe.Pointer(&model))

		return vocab
	}

	if vocabBOSFunc, err = lib.Prep("llama_vocab_bos", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		panic(err)
	}
	VocabBOS = func(vocab Vocab) Token {
		var token ffi.Arg
		vocabBOSFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))

		return Token(token)
	}

	if vocabEOSFunc, err = lib.Prep("llama_vocab_eos", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		panic(err)
	}
	VocabEOS = func(vocab Vocab) Token {
		var token ffi.Arg
		vocabEOSFunc.Call(unsafe.Pointer(&token), unsafe.Pointer(&vocab))

		return Token(token)
	}

	if vocabIsEOGFunc, err = lib.Prep("llama_vocab_is_eog", &ffi.TypeUint8, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		panic(err)
	}
	VocabIsEOG = func(vocab Vocab, token Token) bool {
		var result ffi.Arg
		vocabIsEOGFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&vocab), unsafe.Pointer(&token))

		return result.Bool()
	}

	if vocabNTokensFunc, err = lib.Prep("llama_vocab_n_tokens", &ffi.TypeSint32, &ffi.TypePointer); err != nil {
		panic(err)
	}
	VocabNTokens = func(vocab Vocab) int32 {
		var result ffi.Arg
		vocabNTokensFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&vocab))

		return int32(result)
	}

	if tokenToPieceFunc, err = lib.Prep("llama_token_to_piece", &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32,
		&ffi.TypePointer, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeUint8); err != nil {
		panic(err)
	}
	TokenToPiece = func(vocab Vocab, token Token, buf []byte, lstrip int32, special bool) int32 {
		piece := make([]byte, len(buf)+1)
		b := unsafe.SliceData(piece)
		bLen := int32(len(piece))

		var result ffi.Arg
		tokenToPieceFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&vocab), unsafe.Pointer(&token), unsafe.Pointer(&b),
			&bLen, &lstrip, &special)

		copy(buf, piece)

		return int32(result)
	}

	if tokenizeFunc, err = lib.Prep("llama_tokenize", &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypeSint32,
		&ffi.TypePointer, &ffi.TypeSint32, &ffi.TypeUint8, &ffi.TypeUint8); err != nil {
		panic(err)
	}
	Tokenize = func(vocab Vocab, text string, tokens []Token, addSpecial bool, parseSpecial bool) int32 {
		txt, _ := unix.BytePtrFromString(text)
		txtLen := int32(len(text))

		toks := unsafe.SliceData(tokens)
		nTokensMax := len(tokens)

		var result ffi.Arg
		tokenizeFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&vocab), unsafe.Pointer(&txt), &txtLen,
			unsafe.Pointer(&toks), &nTokensMax, &addSpecial, &parseSpecial)

		// for whatever reason, llama.cpp returns a negative number.
		return -int32(result)
	}
}
