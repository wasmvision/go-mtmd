package llama

import (
	"testing"
	"unsafe"

	"github.com/wasmvision/yzma/pkg/loader"
	"golang.org/x/sys/unix"
)

func TestChat(t *testing.T) {
	testSetup(t)
	defer testCleanup(t)

	role, _ := unix.BytePtrFromString("user")
	content, _ := unix.BytePtrFromString("what is going on?")

	chat := []ChatMessage{ChatMessage{Role: role, Content: content}}
	data := make([]byte, 1024)
	buf := unsafe.SliceData(data)

	sz := ChatApplyTemplate("chatml", unsafe.SliceData(chat), 1, false, buf, int32(len(data)))
	if sz <= 0 {
		t.Fatal("unable to apply chat template")
	}

	result := unix.BytePtrToString(buf)
	if len(result) == 0 {
		t.Fatal("invalid output from chat template")
	}
}

func testSetup(t *testing.T) {
	lib := loader.LoadLibrary("../../lib")
	Load(lib)

	BackendInit()
}

func testCleanup(t *testing.T) {
	BackendFree()
}
