package llama

import (
	"unsafe"

	"github.com/jupiterrider/ffi"
	"golang.org/x/sys/unix"
)

var (
	// LLAMA_API int32_t llama_chat_apply_template(
	//                         const char * tmpl,
	//    const struct llama_chat_message * chat,
	//                             size_t   n_msg,
	//                               bool   add_ass,
	//                               char * buf,
	//                            int32_t   length);
	ChatApplyTemplate     func(tmpl string, chat *ChatMessage, nMsg uint32, addAss bool, buf *byte, len int32) int32
	chatApplyTemplateFunc ffi.Fun
)

func loadChatFuncs(lib ffi.Lib) {
	var err error
	if chatApplyTemplateFunc, err = lib.Prep("llama_chat_apply_template", &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypeUint32,
		&ffi.TypeUint8, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		panic(err)
	}
	ChatApplyTemplate = func(template string, chat *ChatMessage, nMsg uint32, addAss bool, buf *byte, len int32) int32 {
		var result ffi.Arg
		tmpl, _ := unix.BytePtrFromString(template)

		chatApplyTemplateFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&tmpl), unsafe.Pointer(&chat), &nMsg, &addAss, unsafe.Pointer(&buf), &len)
		return int32(result)
	}
}
