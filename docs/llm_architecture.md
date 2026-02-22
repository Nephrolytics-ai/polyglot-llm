# LLM Architecture Design

## Goals

- Keep a single provider-agnostic abstraction for content generation, embeddings, and audio transcription.
- Keep provider implementations isolated under `pkg/llms/*`.
- Support local tools and MCP tools.
- Support prompt context accumulation from static messages and runtime providers.
- Return normalized metadata for observability.
- Return wrapped errors; never use `panic` or `fatal`.

## Base Abstraction Layer (`pkg/model/llm.go`, `pkg/model/embedding.go`, `pkg/model/audio.go`)

This is the contract layer all providers implement.

### Factory Function Types

- `NewStructureContentGeneratorFunc[T any]`
- `NewStringContentGeneratorFunc`
- `NewEmbeddingGeneratorFunc`
- `NewAudioTranscriptionGeneratorFunc`

### Core Interfaces

- `ContentGenerator[T]`
  - `Generate(ctx context.Context) (T, GenerationMetadata, error)`
  - `AddPromptContext(ctx context.Context, messageType ContextMessageType, content string)`
  - `AddPromptContextProvider(ctx context.Context, provider PromptContextProvider)`
- `EmbeddingGenerator`
  - `Generate(ctx context.Context, input string) (EmbeddingVector, GenerationMetadata, error)`
  - `GenerateBatch(ctx context.Context, inputs []string) (EmbeddingVectors, GenerationMetadata, error)`
- `AudioTranscriptionGenerator`
  - `Generate(ctx context.Context) (string, GenerationMetadata, error)`

### Prompt Context Model

- `PromptContext` has:
  - `MessageType` (`system`, `human`, `assistant`)
  - `Content`
- `PromptContextProvider`:
  - `GenerateContext(ctx context.Context) ([]*PromptContext, error)`

Each generator merges:
- static context from `AddPromptContext`
- dynamic context from `AddPromptContextProvider`

### Unified Options Model

All options are `GeneratorOption` and resolve into `GeneratorConfig`:

- `WithIgnoreInvalidGeneratorOptions(bool)`
- `WithURL(string)`
- `WithAuthToken(string)`
- `WithTemperature(float64)`
- `WithMaxTokens(int)`
- `WithEmbeddingDimensions(int)`
- `WithModel(string)`
- `WithReasoningLevel(ReasoningLevel)` where level is `none|low|med|high`
- `WithTools([]Tool)`
- `WithMCPTools([]MCPTool)`

Audio-specific options are passed with `model.AudioOptions`:

- `Prompt string`
- `Keywords []model.AudioKeyword`

Keyword prompt quirk:

- If `AudioOptions.Prompt` is provided, providers use it directly and do not append keyword hints.
- If `AudioOptions.Prompt` is empty and keywords are provided, providers may append:
  - `Common missed words: <json-array-of-audio-keywords>`

### Tools and MCP Tools

- `Tool`
  - `Name`
  - `Description`
  - `InputSchema` (`JSONSchema`)
  - `Handler func(ctx context.Context, args json.RawMessage) (any, error)`
- `MCPTool`
  - `URL`
  - `Name`
  - `HTTPHeaders`
  - `AllowedTools`

### Metadata Contract

`GenerationMetadata` is `map[string]string`.

Common keys:
- `provider`
- `model`
- `latency_ms`
- `input_tokens`
- `output_tokens`
- `total_tokens`
- `cached_input_tokens`
- `reasoning_tokens`
- `api_calls`
- `tool_rounds`
- `response_id`
- `response_status`
- `embedding_count`
- `embedding_dims`

Providers may add additional keys, but these should remain stable.

## Provider Matrix

| Provider | Package | Structured/String Generation | Embeddings | Auth | URL Configuration | Internal APIs Used | MCP Support Mode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OpenAI Responses | `pkg/llms/openai` | Yes | Yes | `WithAuthToken`; if omitted, `openai-go` can read `OPENAI_API_KEY` | `WithURL` -> OpenAI client base URL | `openai-go/v3`: `Responses.New`, `Embeddings.New` | Native MCP via OpenAI Responses MCP tool type |
| Gemini | `pkg/llms/gemini` | Yes | Yes | `WithAuthToken` or env `GEMINI_KEY` | `WithURL` -> `genai.HTTPOptions.BaseURL` | `google.golang.org/genai`: `Models.GenerateContent`, `Models.EmbedContent` | Uses MCP Tool Adapter (`pkg/mcp`) to bridge MCP into normal tool calls |
| Bedrock | `pkg/llms/bedrock` | Yes | No | Env only: `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (optional `AWS_SESSION_TOKEN`) OR `AWS_PROFILE`; region from `AWS_REGION` (default `us-east-1`) | `WithURL` -> Bedrock `BaseEndpoint` override | `aws-sdk-go-v2/service/bedrockruntime`: `Converse` | Uses MCP Tool Adapter (`pkg/mcp`) to bridge MCP into normal tool calls |
| Ollama | `pkg/llms/ollama` | Yes | Yes | None required | `WithURL`, else `OLLAMA_BASE_URL`, else `http://localhost:11434` | Native HTTP `/api/chat` (including tool loop), `/api/embed` with fallback `/api/embeddings` | Uses MCP Tool Adapter (`pkg/mcp`) to bridge MCP into normal tool calls |
| HuggingFace | `pkg/llms/huggingface` | Yes | Yes | `WithAuthToken` or env `HF_TOKEN` | `WithURL`, else `HF_BASE_URL`, else `https://router.huggingface.co` | Raw HTTP: `/v1/chat/completions` (OpenAI-compatible) for generation, `/hf-inference/models/{model}` (native HF feature-extraction) for embeddings | Uses MCP Tool Adapter (`pkg/mcp`) to bridge MCP into normal tool calls |
| Anthopic (scaffold) | `pkg/llms/anthopic` | Constructors exist; `Generate` currently returns not-implemented errors | No | Not implemented | Not implemented | Not implemented | Not implemented |

## OpenAI Responses Details

- Uses structured input items (`ResponseInputItem`) with explicit message roles.
- Supports local tools (`function`) and native OpenAI MCP tools in the same request.
- Implements a stateless tool loop:
  - appends prior model output items into local input history
  - executes tool calls locally
  - appends `function_call_output` items
  - resubmits full history each round
- Does not rely on `previous_response_id`, which keeps it compatible with Zero Data Retention org restrictions.
- Structured generation uses strict JSON schema from `invopop/jsonschema`.
- Applies reasoning/temperature compatibility checks by model family, with optional ignore behavior via `WithIgnoreInvalidGeneratorOptions(true)`.

## Gemini Details

- Uses `genai.Client` with `BackendGeminiAPI`.
- Supports:
  - string generation
  - structured generation (schema-mode when no tools; prompt-enforced JSON when tools are enabled)
  - single and batch embeddings
- Function calling is enabled via Gemini function declarations and tool config.
- MCP tools are converted to local tools through `pkg/mcp.ToolAdapter`.
- Includes fallback logic for models that reject explicit thinking level.

## Bedrock Details

- Uses Bedrock `Converse` API for generation.
- Supports local tools through Bedrock `ToolConfiguration`.
- MCP tools are converted into local tools through `pkg/mcp.ToolAdapter`.
- Supports `WithTemperature` and `WithMaxTokens` mapping into Bedrock inference config.
- Embeddings are not implemented in this provider yet.

## Ollama Details

- Uses native `/api/chat` request/response handling for both normal generation and tool rounds.
- Maintains assistant/tool context history in-process for multi-round tool calling.
- Accepts native `tool_calls` from the model and executes mapped handlers.
- Handles model-side tool name prefixes (for example `tool.<name>`) when resolving handlers.
- Embeddings use `/api/embed`; fallback to `/api/embeddings` for older Ollama servers.
- MCP tools are converted into local tools through `pkg/mcp.ToolAdapter`.

## HuggingFace Details

- Uses raw HTTP against HuggingFace's `router.huggingface.co` (no external SDK dependency).
- Content generation (string, structured, tool calling) uses the OpenAI-compatible `/v1/chat/completions` endpoint.
- Embeddings use the native HF Inference API feature-extraction pipeline at `/hf-inference/models/{model}`.
  - Response parsing handles multiple formats: 2D arrays (sentence-level from TEI-served models), 1D arrays (single input edge case), and 3D arrays (token-level from raw transformer models, mean-pooled to sentence vectors).
- Default generation model: `Qwen/Qwen2.5-72B-Instruct`. Default embedding model: `BAAI/bge-base-en-v1.5`.
- Supports `WithTemperature` and `WithMaxTokens`. `WithReasoningLevel` is not supported (returns error or warns depending on `WithIgnoreInvalidGeneratorOptions`).
- MCP tools are converted into local tools through `pkg/mcp.ToolAdapter`.
- Audio transcription is not supported (returns unsupported error).

## MCP Tool Adapter (`pkg/mcp`)

Providers that do not support MCP natively (Gemini, Bedrock, Ollama, HuggingFace) use `ToolAdapter`:

- Connect to MCP server via streamable HTTP transport.
- Initialize and list tools.
- Convert MCP tool definitions into `model.Tool` entries.
- Execute MCP tool calls through adapter handlers.
- Optional allow-list filtering via `AllowedTools`.

The tool-name cache helper in `pkg/mcp/tools.go` caches per MCP URL.
