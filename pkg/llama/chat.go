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
	ChatApplyTemplate     func(tmpl string, chat []ChatMessage, addAss bool, buf []byte) int32
	chatApplyTemplateFunc ffi.Fun
)

func loadChatFuncs(lib ffi.Lib) {
	var err error
	if chatApplyTemplateFunc, err = lib.Prep("llama_chat_apply_template", &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypeUint32,
		&ffi.TypeUint8, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		panic(err)
	}
	ChatApplyTemplate = func(template string, chat []ChatMessage, addAss bool, buf []byte) int32 {
		tmpl, _ := unix.BytePtrFromString(template)

		c := unsafe.SliceData(chat)
		nMsg := uint32(len(chat))

		out := unsafe.SliceData(buf)
		len := uint32(len(buf))

		var result ffi.Arg
		chatApplyTemplateFunc.Call(unsafe.Pointer(&result), unsafe.Pointer(&tmpl), unsafe.Pointer(&c), &nMsg, &addAss, unsafe.Pointer(&out), &len)
		return int32(result)
	}
}

func NewChatMessage(role, content string) ChatMessage {
	r, _ := unix.BytePtrFromString(role)
	c, _ := unix.BytePtrFromString(content)

	return ChatMessage{Role: r, Content: c}
}
