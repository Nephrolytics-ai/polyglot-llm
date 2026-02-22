package model

import (
	"context"
	"encoding/json"
)

// Provider implementation notes:
//
// Every LLM provider package should expose factory functions matching these
// signatures (for example NewStringContentGenerator and
// NewStructureContentGenerator[T]). This keeps provider wiring consistent and
// allows callers to switch providers without changing calling patterns.
//
// Recommended provider behavior:
//   - Validate required inputs in constructors (for example prompt must not be blank).
//   - Resolve options once via ResolveGeneratorOpts(opts...).
//   - If an option is unsupported:
//   - Return an error by default.
//   - When IgnoreInvalidGeneratorOptions is true, ignore unsupported options.
//   - Return GenerationMetadata with stable keys from this package when available.
//   - Keep provider-specific implementation details inside provider packages.
//

// NewStructureContentGeneratorFunc is for generators that produce structured output (i.e. JSON that can be unmarshaled into a struct).
type NewStructureContentGeneratorFunc[T any] func(prompt string, opts ...GeneratorOption) (ContentGenerator[T], error)

// NewStringContentGeneratorFunc is for generators that produce simple string output.
type NewStringContentGeneratorFunc func(prompt string, opts ...GeneratorOption) (ContentGenerator[string], error)

// NewEmbeddingGeneratorFunc creates an embedding generator.
// Inputs are provided at Generate / GenerateBatch call time.
type NewEmbeddingGeneratorFunc func(opts ...GeneratorOption) (EmbeddingGenerator, error)

// NewAudioTranscriptionGeneratorFunc creates an audio transcription generator for a source file.
type NewAudioTranscriptionGeneratorFunc func(filePath string, opts AudioOptions) (AudioTranscriptionGenerator, error)

type ContentGenerator[T any] interface {
	Generate(ctx context.Context) (T, GenerationMetadata, error)
	AddPromptContext(ctx context.Context, messageType ContextMessageType, content string)
	AddPromptContextProvider(ctx context.Context, provider PromptContextProvider)
}

type EmbeddingGenerator interface {
	Generate(ctx context.Context, input string) (EmbeddingVector, GenerationMetadata, error)
	GenerateBatch(ctx context.Context, inputs []string) (EmbeddingVectors, GenerationMetadata, error)
}

// AudioTranscriptionGenerator represents "audio file in, transcript out".
type AudioTranscriptionGenerator interface {
	Generate(ctx context.Context) (string, GenerationMetadata, error)
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

// GeneratorConfig is the resolved internal representation of GeneratorOption values.
//
// Field semantics:
//   - IgnoreInvalidGeneratorOptions: ignore unsupported options instead of returning an error.
//   - URL: override provider endpoint/base URL.
//   - AuthToken: override provider API token/auth value.
//   - Temperature: optional sampling temperature for text generation.
//   - MaxTokens: optional output token limit for text generation.
//   - EmbeddingDimensions: optional embedding size where provider supports it.
//   - Model: optional explicit model name override.
//   - ReasoningLevel: optional reasoning effort level for models that support it.
//   - Tools: optional local function/tool declarations and handlers.
//   - MCPTools: optional remote MCP tool servers to expose during generation.
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
	URL  string
	Name string
	// AuthToken is used by providers that require MCP auth outside HTTPHeaders (for example, Anthropic authorization_token).
	AuthToken   string
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

// WithIgnoreInvalidGeneratorOptions configures whether providers should ignore
// unsupported options instead of returning errors.
func WithIgnoreInvalidGeneratorOptions(value bool) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.IgnoreInvalidGeneratorOptions = value
	})
}

// WithURL sets a provider-specific base URL/endpoint override.
func WithURL(value string) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.URL = value
	})
}

// WithAuthToken sets a provider-specific API auth token override.
func WithAuthToken(value string) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.AuthToken = value
	})
}

// WithTemperature sets generation sampling temperature when supported.
func WithTemperature(value float64) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.Temperature = &value
	})
}

// WithMaxTokens sets max output tokens when supported.
func WithMaxTokens(value int) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.MaxTokens = &value
	})
}

// WithModel sets an explicit model name.
func WithModel(value string) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.Model = &value
	})
}

// WithTools sets local tool/function declarations for tool calling.
func WithTools(tools []Tool) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.Tools = append([]Tool(nil), tools...)
	})
}

// WithMCPTools sets MCP tool server declarations.
func WithMCPTools(tools []MCPTool) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.MCPTools = append([]MCPTool(nil), tools...)
	})
}

// WithReasoningLevel sets reasoning effort for models/providers that support it.
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
