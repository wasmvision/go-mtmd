package llama

// Common types matching llama.cpp
type (
	Token  int32
	Pos    int32
	SeqId  int32
	Memory uintptr
)

// Constants from llama.h
const (
	DEFAULT_SEED = 0xFFFFFFFF
	TOKEN_NULL   = -1

	// File magic numbers
	FILE_MAGIC_GGLA = 0x67676c61
	FILE_MAGIC_GGSN = 0x6767736e
	FILE_MAGIC_GGSQ = 0x67677371

	// Session constants
	SESSION_MAGIC   = FILE_MAGIC_GGSN
	SESSION_VERSION = 9

	STATE_SEQ_MAGIC   = FILE_MAGIC_GGSQ
	STATE_SEQ_VERSION = 2
)

// Enums
type VocabType int32

const (
	VOCAB_TYPE_NONE VocabType = iota
	VOCAB_TYPE_SPM
	VOCAB_TYPE_BPE
	VOCAB_TYPE_WPM
	VOCAB_TYPE_UGM
	VOCAB_TYPE_RWKV
)

type TokenType int32

const (
	TOKEN_TYPE_UNDEFINED TokenType = iota
	TOKEN_TYPE_NORMAL
	TOKEN_TYPE_UNKNOWN
	TOKEN_TYPE_CONTROL
	TOKEN_TYPE_USER_DEFINED
	TOKEN_TYPE_UNUSED
	TOKEN_TYPE_BYTE
)

type TokenAttr int32

const (
	TOKEN_ATTR_UNDEFINED   TokenAttr = 0
	TOKEN_ATTR_UNKNOWN     TokenAttr = 1 << 0
	TOKEN_ATTR_UNUSED      TokenAttr = 1 << 1
	TOKEN_ATTR_NORMAL      TokenAttr = 1 << 2
	TOKEN_ATTR_CONTROL     TokenAttr = 1 << 3
	TOKEN_ATTR_USER_DEF    TokenAttr = 1 << 4
	TOKEN_ATTR_BYTE        TokenAttr = 1 << 5
	TOKEN_ATTR_LSTRIP      TokenAttr = 1 << 6
	TOKEN_ATTR_RSTRIP      TokenAttr = 1 << 7
	TOKEN_ATTR_SINGLE_WORD TokenAttr = 1 << 8
)

type Ftype int32

const (
	FTYPE_ALL_F32        Ftype = 0
	FTYPE_MOSTLY_F16     Ftype = 1
	FTYPE_MOSTLY_Q4_0    Ftype = 2
	FTYPE_MOSTLY_Q4_1    Ftype = 3
	FTYPE_MOSTLY_Q8_0    Ftype = 7
	FTYPE_MOSTLY_Q5_0    Ftype = 8
	FTYPE_MOSTLY_Q5_1    Ftype = 9
	FTYPE_MOSTLY_Q2_K    Ftype = 10
	FTYPE_MOSTLY_Q3_K_S  Ftype = 11
	FTYPE_MOSTLY_Q3_K_M  Ftype = 12
	FTYPE_MOSTLY_Q3_K_L  Ftype = 13
	FTYPE_MOSTLY_Q4_K_S  Ftype = 14
	FTYPE_MOSTLY_Q4_K_M  Ftype = 15
	FTYPE_MOSTLY_Q5_K_S  Ftype = 16
	FTYPE_MOSTLY_Q5_K_M  Ftype = 17
	FTYPE_MOSTLY_Q6_K    Ftype = 18
	FTYPE_MOSTLY_IQ2_XXS Ftype = 19
	FTYPE_MOSTLY_IQ2_XS  Ftype = 20
	FTYPE_MOSTLY_Q2_K_S  Ftype = 21
	FTYPE_MOSTLY_IQ3_XS  Ftype = 22
)

type RopeScalingType int32

const (
	ROPE_SCALING_TYPE_UNSPECIFIED RopeScalingType = -1
	ROPE_SCALING_TYPE_NONE        RopeScalingType = 0
	ROPE_SCALING_TYPE_LINEAR      RopeScalingType = 1
	ROPE_SCALING_TYPE_YARN        RopeScalingType = 2
)

type PoolingType int32

const (
	POOLING_TYPE_UNSPECIFIED PoolingType = -1
	POOLING_TYPE_NONE        PoolingType = 0
	POOLING_TYPE_MEAN        PoolingType = 1
	POOLING_TYPE_CLS         PoolingType = 2
	POOLING_TYPE_LAST        PoolingType = 3
	POOLING_TYPE_RANK        PoolingType = 4
)

type AttentionType int32

const (
	ATTENTION_TYPE_CAUSAL     AttentionType = 0
	ATTENTION_TYPE_NON_CAUSAL AttentionType = 1
)

type FlashAttentionType int32

const (
	LLAMA_FLASH_ATTN_TYPE_AUTO     FlashAttentionType = -1
	LLAMA_FLASH_ATTN_TYPE_DISABLED FlashAttentionType = 0
	LLAMA_FLASH_ATTN_TYPE_ENABLED  FlashAttentionType = 1
)

type SplitMode int32

const (
	SPLIT_MODE_NONE  SplitMode = 0
	SPLIT_MODE_LAYER SplitMode = 1
	SPLIT_MODE_ROW   SplitMode = 2
)

type GpuBackend int32

const (
	GPU_BACKEND_NONE   GpuBackend = 0
	GPU_BACKEND_CPU    GpuBackend = 1
	GPU_BACKEND_CUDA   GpuBackend = 2
	GPU_BACKEND_METAL  GpuBackend = 3
	GPU_BACKEND_HIP    GpuBackend = 4
	GPU_BACKEND_VULKAN GpuBackend = 5
	GPU_BACKEND_OPENCL GpuBackend = 6
	GPU_BACKEND_SYCL   GpuBackend = 7
)

// String returns the string representation of the GPU backend
func (b GpuBackend) String() string {
	switch b {
	case GPU_BACKEND_NONE:
		return "None"
	case GPU_BACKEND_CPU:
		return "CPU"
	case GPU_BACKEND_CUDA:
		return "CUDA"
	case GPU_BACKEND_METAL:
		return "Metal"
	case GPU_BACKEND_HIP:
		return "HIP"
	case GPU_BACKEND_VULKAN:
		return "Vulkan"
	case GPU_BACKEND_OPENCL:
		return "OpenCL"
	case GPU_BACKEND_SYCL:
		return "SYCL"
	default:
		return "Unknown"
	}
}

type LogLevel int32

const (
	LogLevelNone     LogLevel = 0
	LogLevelDebug    LogLevel = 1
	LogLevelInfo     LogLevel = 2
	LogLevelWarn     LogLevel = 3
	LogLevelError    LogLevel = 4
	LogLevelContinue LogLevel = 5
)

// Opaque types (represented as pointers)
type Model uintptr
type Context uintptr
type Vocab uintptr
type AdapterLora uintptr

// Structs
type TokenData struct {
	Id    Token   // token id
	Logit float32 // log-odds of the token
	P     float32 // probability of the token
}

type TokenDataArray struct {
	Data     *TokenData // pointer to token data array
	Size     uint64     // number of tokens
	Selected int64      // index of selected token (-1 if none)
	Sorted   uint8      // whether the array is sorted by probability (bool as uint8)
}

type Batch struct {
	NTokens int32    // number of tokens
	Token   *Token   // tokens
	Embd    *float32 // embeddings (if using embeddings instead of tokens)
	Pos     *Pos     // positions
	NSeqId  *int32   // number of sequence IDs per token
	SeqId   **SeqId  // sequence IDs
	Logits  *int8    // whether to compute logits for each token
}

// Model parameters
type ModelParams struct {
	Devices                  uintptr   // ggml_backend_dev_t * - NULL-terminated list of devices
	TensorBuftOverrides      uintptr   // const struct llama_model_tensor_buft_override *
	NGpuLayers               int32     // number of layers to store in VRAM
	SplitMode                SplitMode // how to split the model across multiple GPUs
	MainGpu                  int32     // the GPU that is used for the entire model
	TensorSplit              *float32  // proportion of the model to offload to each GPU
	ProgressCallback         uintptr   // llama_progress_callback function pointer
	ProgressCallbackUserData uintptr   // context pointer passed to the progress callback
	KvOverrides              uintptr   // const struct llama_model_kv_override *
	VocabOnly                uint8     // only load the vocabulary, no weights (bool as uint8)
	UseMmap                  uint8     // use mmap if possible (bool as uint8)
	UseMlock                 uint8     // force system to keep model in RAM (bool as uint8)
	CheckTensors             uint8     // validate model tensor data (bool as uint8)
	UseExtraBufts            uint8     // use extra buffer types (bool as uint8)
}

// Context parameters
type ContextParams struct {
	NCtx               uint32             // text context, 0 = from model
	NBatch             uint32             // logical maximum batch size
	NUbatch            uint32             // physical maximum batch size
	NSeqMax            uint32             // max number of sequences
	NThreads           int32              // number of threads to use for generation
	NThreadsBatch      int32              // number of threads to use for batch processing
	RopeScalingType    RopeScalingType    // RoPE scaling type
	PoolingType        PoolingType        // pooling type for embeddings
	AttentionType      AttentionType      // attention type
	FlashAttentionType FlashAttentionType // when to enable Flash Attention
	RopeFreqBase       float32            // RoPE base frequency
	RopeFreqScale      float32            // RoPE frequency scaling factor
	YarnExtFactor      float32            // YaRN extrapolation mix factor
	YarnAttnFactor     float32            // YaRN magnitude scaling factor
	YarnBetaFast       float32            // YaRN low correction dim
	YarnBetaSlow       float32            // YaRN high correction dim
	YarnOrigCtx        uint32             // YaRN original context size
	DefragThold        float32            // defragment the KV cache if holes/size > thold
	CbEval             uintptr            // evaluation callback
	CbEvalUserData     uintptr            // user data for evaluation callback
	TypeK              int32              // data type for K cache
	TypeV              int32              // data type for V cache
	AbortCallback      uintptr            // abort callback
	AbortCallbackData  uintptr            // user data for abort callback
	Embeddings         uint8              // whether to compute and return embeddings (bool as uint8)
	Offload_kqv        uint8              // whether to offload K, Q, V to GPU (bool as uint8)
	NoPerf             uint8              // whether to measure performance (bool as uint8)
	OpOffload          uint8              // offload host tensor operations to device
	SwaFull            uint8              // use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
	KVUnified          uint8              // use a unified buffer across the input sequences when computing the attentions
}

// Model quantize parameters
type ModelQuantizeParams struct {
	NThread              int32 // number of threads to use for quantizing
	Ftype                Ftype // quantize to this llama_ftype
	OutputTensorType     int32 // output tensor type
	TokenEmbeddingType   int32 // itoken embeddings tensor type
	AllowRequantize      uint8 // allow quantizing non-f32/f16 tensors (bool as uint8)
	QuantizeOutputTensor uint8 // quantize output.weight (bool as uint8)
	OnlyF32              uint8 // quantize only f32 tensors (bool as uint8)
	PureF16              uint8 // disable k-quant mixtures and quantize all tensors to the same type (bool as uint8)
	KeepSplit            uint8 // keep split tensors (bool as uint8)
	IMatrix              *byte // importance matrix data
	KqsWarning           uint8 // warning for quantization quality loss (bool as uint8)
}

// Chat message
type ChatMessage struct {
	Role    *byte // role string
	Content *byte // content string
}

// Sampler chain parameters
type SamplerChainParams struct {
	NoPerf uint8 // whether to measure performance timings (bool as uint8)
}

// Logit bias
type LogitBias struct {
	Token Token
	Bias  float32
}
