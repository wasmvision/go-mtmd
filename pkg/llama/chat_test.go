package llama

import (
	"testing"
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
