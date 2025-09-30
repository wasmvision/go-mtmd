package llama

import "github.com/jupiterrider/ffi"

func Load(lib ffi.Lib) error {

	if err := loadFuncs(lib); err != nil {
		return err
	}

	if err := loadModelFuncs(lib); err != nil {
		return err
	}

	if err := loadBatchFuncs(lib); err != nil {
		return err
	}

	if err := loadVocabFuncs(lib); err != nil {
		return err
	}

	if err := loadSamplingFuncs(lib); err != nil {
		return err
	}

	if err := loadChatFuncs(lib); err != nil {
		return err
	}

	if err := loadContextFuncs(lib); err != nil {
		return err
	}

	if err := loadLogFuncs(lib); err != nil {
		return err
	}

	return nil
}
