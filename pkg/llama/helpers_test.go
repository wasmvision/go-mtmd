package llama

import (
	"testing"

	"github.com/hybridgroup/yzma/pkg/loader"
)

func testSetup(t *testing.T) {
	testPath := "."
	lib, err := loader.LoadLibrary(testPath)
	if err != nil {
		t.Fatal("unable to load library", err.Error())
	}
	if err := Load(lib); err != nil {
		t.Fatal("unable to load library", err.Error())
	}

	BackendInit()
	GGMLBackendLoadAll()
}

func testCleanup(t *testing.T) {
	BackendFree()
}
