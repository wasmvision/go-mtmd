package llama

import (
	"testing"

	"github.com/wasmvision/yzma/pkg/loader"
)

func TestChat(t *testing.T) {
	testSetup(t)
	defer testCleanup(t)

	chat := []ChatMessage{NewChatMessage("user", "what is going on?")}
	buf := make([]byte, 1024)

	sz := ChatApplyTemplate("chatml", chat, false, buf)
	if sz <= 0 {
		t.Fatal("unable to apply chat template")
	}

	result := string(buf)
	if len(result) == 0 {
		t.Fatal("invalid output from chat template")
	}
}

func testSetup(t *testing.T) {
	lib, err := loader.LoadLibrary("../../lib")
	if err != nil {
		t.Fatal("unable to load libary", err.Error())
	}
	Load(lib)

	BackendInit()
}

func testCleanup(t *testing.T) {
	BackendFree()
}
