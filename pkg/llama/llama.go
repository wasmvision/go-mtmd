package llama

// Common types matching llama.cpp
type (
	LlamaToken  int32
	LlamaPos    int32
	LlamaSeqId  int32
	LlamaMemory uintptr
)

// Constants from llama.h
const (
	LLAMA_DEFAULT_SEED = 0xFFFFFFFF
	LLAMA_TOKEN_NULL   = -1

	// File magic numbers
	LLAMA_FILE_MAGIC_GGLA = 0x67676c61
	LLAMA_FILE_MAGIC_GGSN = 0x6767736e
	LLAMA_FILE_MAGIC_GGSQ = 0x67677371

	// Session constants
	LLAMA_SESSION_MAGIC   = LLAMA_FILE_MAGIC_GGSN
	LLAMA_SESSION_VERSION = 9

	LLAMA_STATE_SEQ_MAGIC   = LLAMA_FILE_MAGIC_GGSQ
	LLAMA_STATE_SEQ_VERSION = 2
)

// Enums
type LlamaVocabType int32

const (
	LLAMA_VOCAB_TYPE_NONE LlamaVocabType = iota
	LLAMA_VOCAB_TYPE_SPM
	LLAMA_VOCAB_TYPE_BPE
	LLAMA_VOCAB_TYPE_WPM
	LLAMA_VOCAB_TYPE_UGM
	LLAMA_VOCAB_TYPE_RWKV
)

type LlamaTokenType int32

const (
	LLAMA_TOKEN_TYPE_UNDEFINED LlamaTokenType = iota
	LLAMA_TOKEN_TYPE_NORMAL
	LLAMA_TOKEN_TYPE_UNKNOWN
	LLAMA_TOKEN_TYPE_CONTROL
	LLAMA_TOKEN_TYPE_USER_DEFINED
	LLAMA_TOKEN_TYPE_UNUSED
	LLAMA_TOKEN_TYPE_BYTE
)

type LlamaTokenAttr int32

const (
	LLAMA_TOKEN_ATTR_UNDEFINED   LlamaTokenAttr = 0
	LLAMA_TOKEN_ATTR_UNKNOWN     LlamaTokenAttr = 1 << 0
	LLAMA_TOKEN_ATTR_UNUSED      LlamaTokenAttr = 1 << 1
	LLAMA_TOKEN_ATTR_NORMAL      LlamaTokenAttr = 1 << 2
	LLAMA_TOKEN_ATTR_CONTROL     LlamaTokenAttr = 1 << 3
	LLAMA_TOKEN_ATTR_USER_DEF    LlamaTokenAttr = 1 << 4
	LLAMA_TOKEN_ATTR_BYTE        LlamaTokenAttr = 1 << 5
	LLAMA_TOKEN_ATTR_LSTRIP      LlamaTokenAttr = 1 << 6
	LLAMA_TOKEN_ATTR_RSTRIP      LlamaTokenAttr = 1 << 7
	LLAMA_TOKEN_ATTR_SINGLE_WORD LlamaTokenAttr = 1 << 8
)

type LlamaFtype int32

const (
	LLAMA_FTYPE_ALL_F32        LlamaFtype = 0
	LLAMA_FTYPE_MOSTLY_F16     LlamaFtype = 1
	LLAMA_FTYPE_MOSTLY_Q4_0    LlamaFtype = 2
	LLAMA_FTYPE_MOSTLY_Q4_1    LlamaFtype = 3
	LLAMA_FTYPE_MOSTLY_Q8_0    LlamaFtype = 7
	LLAMA_FTYPE_MOSTLY_Q5_0    LlamaFtype = 8
	LLAMA_FTYPE_MOSTLY_Q5_1    LlamaFtype = 9
	LLAMA_FTYPE_MOSTLY_Q2_K    LlamaFtype = 10
	LLAMA_FTYPE_MOSTLY_Q3_K_S  LlamaFtype = 11
	LLAMA_FTYPE_MOSTLY_Q3_K_M  LlamaFtype = 12
	LLAMA_FTYPE_MOSTLY_Q3_K_L  LlamaFtype = 13
	LLAMA_FTYPE_MOSTLY_Q4_K_S  LlamaFtype = 14
	LLAMA_FTYPE_MOSTLY_Q4_K_M  LlamaFtype = 15
	LLAMA_FTYPE_MOSTLY_Q5_K_S  LlamaFtype = 16
	LLAMA_FTYPE_MOSTLY_Q5_K_M  LlamaFtype = 17
	LLAMA_FTYPE_MOSTLY_Q6_K    LlamaFtype = 18
	LLAMA_FTYPE_MOSTLY_IQ2_XXS LlamaFtype = 19
	LLAMA_FTYPE_MOSTLY_IQ2_XS  LlamaFtype = 20
	LLAMA_FTYPE_MOSTLY_Q2_K_S  LlamaFtype = 21
	LLAMA_FTYPE_MOSTLY_IQ3_XS  LlamaFtype = 22
)

type LlamaRopeScalingType int32

const (
	LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED LlamaRopeScalingType = -1
	LLAMA_ROPE_SCALING_TYPE_NONE        LlamaRopeScalingType = 0
	LLAMA_ROPE_SCALING_TYPE_LINEAR      LlamaRopeScalingType = 1
	LLAMA_ROPE_SCALING_TYPE_YARN        LlamaRopeScalingType = 2
)

type LlamaPoolingType int32

const (
	LLAMA_POOLING_TYPE_UNSPECIFIED LlamaPoolingType = -1
	LLAMA_POOLING_TYPE_NONE        LlamaPoolingType = 0
	LLAMA_POOLING_TYPE_MEAN        LlamaPoolingType = 1
	LLAMA_POOLING_TYPE_CLS         LlamaPoolingType = 2
	LLAMA_POOLING_TYPE_LAST        LlamaPoolingType = 3
	LLAMA_POOLING_TYPE_RANK        LlamaPoolingType = 4
)

type LlamaAttentionType int32

const (
	LLAMA_ATTENTION_TYPE_CAUSAL     LlamaAttentionType = 0
	LLAMA_ATTENTION_TYPE_NON_CAUSAL LlamaAttentionType = 1
)

type LlamaSplitMode int32

const (
	LLAMA_SPLIT_MODE_NONE  LlamaSplitMode = 0
	LLAMA_SPLIT_MODE_LAYER LlamaSplitMode = 1
	LLAMA_SPLIT_MODE_ROW   LlamaSplitMode = 2
)

type LlamaGpuBackend int32

const (
	LLAMA_GPU_BACKEND_NONE   LlamaGpuBackend = 0
	LLAMA_GPU_BACKEND_CPU    LlamaGpuBackend = 1
	LLAMA_GPU_BACKEND_CUDA   LlamaGpuBackend = 2
	LLAMA_GPU_BACKEND_METAL  LlamaGpuBackend = 3
	LLAMA_GPU_BACKEND_HIP    LlamaGpuBackend = 4
	LLAMA_GPU_BACKEND_VULKAN LlamaGpuBackend = 5
	LLAMA_GPU_BACKEND_OPENCL LlamaGpuBackend = 6
	LLAMA_GPU_BACKEND_SYCL   LlamaGpuBackend = 7
)

// String returns the string representation of the GPU backend
func (b LlamaGpuBackend) String() string {
	switch b {
	case LLAMA_GPU_BACKEND_NONE:
		return "None"
	case LLAMA_GPU_BACKEND_CPU:
		return "CPU"
	case LLAMA_GPU_BACKEND_CUDA:
		return "CUDA"
	case LLAMA_GPU_BACKEND_METAL:
		return "Metal"
	case LLAMA_GPU_BACKEND_HIP:
		return "HIP"
	case LLAMA_GPU_BACKEND_VULKAN:
		return "Vulkan"
	case LLAMA_GPU_BACKEND_OPENCL:
		return "OpenCL"
	case LLAMA_GPU_BACKEND_SYCL:
		return "SYCL"
	default:
		return "Unknown"
	}
}

// Opaque types (represented as pointers)
type LlamaModel uintptr
type LlamaContext uintptr
type LlamaVocab uintptr
type LlamaSampler uintptr
type LlamaAdapterLora uintptr

// Structs
type LlamaTokenData struct {
	Id    LlamaToken // token id
	Logit float32    // log-odds of the token
	P     float32    // probability of the token
}

type LlamaTokenDataArray struct {
	Data     *LlamaTokenData // pointer to token data array
	Size     uint64          // number of tokens
	Selected int64           // index of selected token (-1 if none)
	Sorted   uint8           // whether the array is sorted by probability (bool as uint8)
}

type LlamaBatch struct {
	NTokens int32        // number of tokens
	Token   *LlamaToken  // tokens
	Embd    *float32     // embeddings (if using embeddings instead of tokens)
	Pos     *LlamaPos    // positions
	NSeqId  *int32       // number of sequence IDs per token
	SeqId   **LlamaSeqId // sequence IDs
	Logits  *int8        // whether to compute logits for each token
}

// Model parameters
type LlamaModelParams struct {
	Devices                  uintptr        // ggml_backend_dev_t * - NULL-terminated list of devices
	TensorBuftOverrides      uintptr        // const struct llama_model_tensor_buft_override *
	NGpuLayers               int32          // number of layers to store in VRAM
	SplitMode                LlamaSplitMode // how to split the model across multiple GPUs
	MainGpu                  int32          // the GPU that is used for the entire model
	TensorSplit              *float32       // proportion of the model to offload to each GPU
	ProgressCallback         uintptr        // llama_progress_callback function pointer
	ProgressCallbackUserData uintptr        // context pointer passed to the progress callback
	KvOverrides              uintptr        // const struct llama_model_kv_override *
	VocabOnly                uint8          // only load the vocabulary, no weights (bool as uint8)
	UseMmap                  uint8          // use mmap if possible (bool as uint8)
	UseMlock                 uint8          // force system to keep model in RAM (bool as uint8)
	CheckTensors             uint8          // validate model tensor data (bool as uint8)
	UseExtraBufts            uint8          // use extra buffer types (bool as uint8)
}

// Context parameters
type LlamaContextParams struct {
	Seed              uint32               // RNG seed, -1 for random
	NCtx              uint32               // text context, 0 = from model
	NBatch            uint32               // logical maximum batch size
	NUbatch           uint32               // physical maximum batch size
	NSeqMax           uint32               // max number of sequences
	NThreads          int32                // number of threads to use for generation
	NThreadsBatch     int32                // number of threads to use for batch processing
	RopeScalingType   LlamaRopeScalingType // RoPE scaling type
	PoolingType       LlamaPoolingType     // pooling type for embeddings
	AttentionType     LlamaAttentionType   // attention type
	RopeFreqBase      float32              // RoPE base frequency
	RopeFreqScale     float32              // RoPE frequency scaling factor
	YarnExtFactor     float32              // YaRN extrapolation mix factor
	YarnAttnFactor    float32              // YaRN magnitude scaling factor
	YarnBetaFast      float32              // YaRN low correction dim
	YarnBetaSlow      float32              // YaRN high correction dim
	YarnOrigCtx       uint32               // YaRN original context size
	DefragThold       float32              // defragment the KV cache if holes/size > thold
	CbEval            uintptr              // evaluation callback
	CbEvalUserData    uintptr              // user data for evaluation callback
	TypeK             int32                // data type for K cache
	TypeV             int32                // data type for V cache
	AbortCallback     uintptr              // abort callback
	AbortCallbackData uintptr              // user data for abort callback
	Logits            uint8                // whether to compute and return logits (bool as uint8)
	Embeddings        uint8                // whether to compute and return embeddings (bool as uint8)
	Offload_kqv       uint8                // whether to offload K, Q, V to GPU (bool as uint8)
	FlashAttn         uint8                // whether to use flash attention (bool as uint8)
	NoPerf            uint8                // whether to measure performance (bool as uint8)
}

// Model quantize parameters
type LlamaModelQuantizeParams struct {
	NThread              int32      // number of threads to use for quantizing
	Ftype                LlamaFtype // quantize to this llama_ftype
	OutputTensorType     int32      // output tensor type
	TokenEmbeddingType   int32      // itoken embeddings tensor type
	AllowRequantize      uint8      // allow quantizing non-f32/f16 tensors (bool as uint8)
	QuantizeOutputTensor uint8      // quantize output.weight (bool as uint8)
	OnlyF32              uint8      // quantize only f32 tensors (bool as uint8)
	PureF16              uint8      // disable k-quant mixtures and quantize all tensors to the same type (bool as uint8)
	KeepSplit            uint8      // keep split tensors (bool as uint8)
	IMatrix              *byte      // importance matrix data
	KqsWarning           uint8      // warning for quantization quality loss (bool as uint8)
}

// Chat message
type LlamaChatMessage struct {
	Role    *byte // role string
	Content *byte // content string
}

// Sampler chain parameters
type LlamaSamplerChainParams struct {
	NoPerf uint8 // whether to measure performance timings (bool as uint8)
}

// Logit bias
type LlamaLogitBias struct {
	Token LlamaToken
	Bias  float32
}
