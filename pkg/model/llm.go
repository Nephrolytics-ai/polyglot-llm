package model

import (
	"context"
	"encoding/json"
)

// These are factory methods each llm provider should implement to create content generators.

// NewStructureContentGeneratorFunc is for generators that produce structured output (i.e. JSON that can be unmarshaled into a struct).
type NewStructureContentGeneratorFunc[T any] func(prompt string, opts ...GeneratorOption) (ContentGenerator[T], error)

// NewStringContentGeneratorFunc is for generators that produce simple string output.
type NewStringContentGeneratorFunc func(prompt string, opts ...GeneratorOption) (ContentGenerator[string], error)

// NewEmbeddingGeneratorFunc is for generators that produce a single embedding vector.
type NewEmbeddingGeneratorFunc func(input string, opts ...GeneratorOption) (EmbeddingGenerator, error)

// NewBatchEmbeddingGeneratorFunc is for generators that produce embeddings for multiple inputs.
type NewBatchEmbeddingGeneratorFunc func(inputs []string, opts ...GeneratorOption) (EmbeddingGenerator, error)

type ContentGenerator[T any] interface {
	Generate(ctx context.Context) (T, GenerationMetadata, error)
	AddPromptContext(ctx context.Context, messageType ContextMessageType, content string)
	AddPromptContextProvider(ctx context.Context, provider PromptContextProvider)
}

type EmbeddingVector = []float64
type EmbeddingVectors = [][]float64

type EmbeddingGenerator interface {
	Generate(ctx context.Context) (EmbeddingVector, GenerationMetadata, error)
	GenerateBatch(ctx context.Context) (EmbeddingVectors, GenerationMetadata, error)
}

type GenerationMetadata map[string]string

const (
	MetadataKeyProvider          = "provider"
	MetadataKeyModel             = "model"
	MetadataKeyLatencyMs         = "latency_ms"
	MetadataKeyInputTokens       = "input_tokens"
	MetadataKeyOutputTokens      = "output_tokens"
	MetadataKeyTotalTokens       = "total_tokens"
	MetadataKeyCachedInputTokens = "cached_input_tokens"
	MetadataKeyReasoningTokens   = "reasoning_tokens"
	MetadataKeyAPICalls          = "api_calls"
	MetadataKeyToolRounds        = "tool_rounds"
	MetadataKeyResponseID        = "response_id"
	MetadataKeyResponseStatus    = "response_status"
	MetadataKeyEmbeddingCount    = "embedding_count"
	MetadataKeyEmbeddingDims     = "embedding_dims"
)

type PromptContext struct {
	MessageType ContextMessageType
	Content     string
}
type PromptContextProvider interface {
	GenerateContext(ctx context.Context) ([]*PromptContext, error)
}

type ContextMessageType string

const (
	ContextMessageTypeSystem    ContextMessageType = "system"    //Used to provide instructions or context to the model that is not part of the user input or assistant output.  Such as the desired Persona
	ContextMessageTypeHuman     ContextMessageType = "human"     // Context to the LLM as from a human, but not part of the actual prompt.  For example RAG Content
	ContextMessageTypeAssistant ContextMessageType = "assistant" //Chain responses from the assistant.
)

// Deprecated: use ContentGenerator.
type Generator[T any] = ContentGenerator[T]

type GeneratorOption interface {
	apply(*GeneratorConfig)
}

type generatorOptionFunc func(*GeneratorConfig)

func (f generatorOptionFunc) apply(cfg *GeneratorConfig) {
	f(cfg)
}

// Backward-compatible alias for existing call sites.
type GeneratorOpts = GeneratorOption

type GeneratorConfig struct {
	IgnoreInvalidGeneratorOptions bool
	URL                           string
	AuthToken                     string
	Temperature                   *float64
	MaxTokens                     *int
	EmbeddingDimensions           *int
	Model                         *string
	ReasoningLevel                *ReasoningLevel
	Tools                         []Tool
	MCPTools                      []MCPTool
}

type ReasoningLevel string

const (
	ReasoningLevelNone ReasoningLevel = "none"
	ReasoningLevelLow  ReasoningLevel = "low"
	ReasoningLevelMed  ReasoningLevel = "med"
	ReasoningLevelHigh ReasoningLevel = "high"
)

type JSONSchema map[string]any

type Tool struct {
	Name        string
	Description string
	InputSchema JSONSchema

	// Handler gets raw JSON args (already validated by you if you want),
	// and returns JSON output.
	Handler func(ctx context.Context, args json.RawMessage) (any, error)
}

type MCPTool struct {
	URL         string
	Name        string
	HTTPHeaders map[string]string
	// AllowedTools restricts exposed MCP tools. If omitted, all server tools are discovered and used.
	AllowedTools []string
}

func ResolveGeneratorOpts(opts ...GeneratorOption) GeneratorConfig {
	cfg := GeneratorConfig{}
	for _, opt := range opts {
		if opt != nil {
			opt.apply(&cfg)
		}
	}
	return cfg
}

func WithIgnoreInvalidGeneratorOptions(value bool) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.IgnoreInvalidGeneratorOptions = value
	})
}

func WithURL(value string) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.URL = value
	})
}

func WithAuthToken(value string) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.AuthToken = value
	})
}

func WithTemperature(value float64) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.Temperature = &value
	})
}

func WithMaxTokens(value int) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.MaxTokens = &value
	})
}

func WithEmbeddingDimensions(value int) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.EmbeddingDimensions = &value
	})
}

func WithModel(value string) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.Model = &value
	})
}

func WithTools(tools []Tool) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.Tools = append([]Tool(nil), tools...)
	})
}

func WithMCPTools(tools []MCPTool) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.MCPTools = append([]MCPTool(nil), tools...)
	})
}

func WithReasoningLevel(level ReasoningLevel) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.ReasoningLevel = &level
	})
}

// Deprecated: use WithTemperature.
func Temperature(value float64) GeneratorOption {
	return WithTemperature(value)
}

// Deprecated: use WithMaxTokens.
func MaxTokens(value int) GeneratorOption {
	return WithMaxTokens(value)
}

// Deprecated: use WithTools.
func Tools(tools []Tool) GeneratorOption {
	return WithTools(tools)
}
