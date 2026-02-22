# polyglot-llm
# Purpose
This is an open source LLM wrapper library in Go. 
It provides a common interface for interacting with different LLM providers, such as OpenAI, Anthropic, and Azure. The goal is to make it easy to switch between different LLM providers without having to change your code.

This software is provided open by Nephrolytics and is intended to be used by anyone who wants to use LLMs in their projects. We welcome contributions and feedback from the community.

Nephrolytics does NOT provide support on this library, but we will do our best to review and merge pull requests in a timely manner and answer questions as we can.


There is a a library call LangchainGo.  The library is old and not being actively maintained, and it also has a little different design philosophy than what we want to achieve with this library.
But its a good reference for how to implement the LLM interface and the generator interface, as well as how to handle tools and logging.

## How We Handle LLMs
Each provider package in `pkg/llms/<provider>` exposes factory functions that return shared interfaces from `pkg/model`. This keeps your app code provider-agnostic while still letting each provider implement provider-specific behavior.

### Content Generators
For text and structured generation, providers expose:

- `NewStringContentGenerator(prompt string, opts ...model.GeneratorOption) (model.ContentGenerator[string], error)`
- `NewStructureContentGenerator[T any](prompt string, opts ...model.GeneratorOption) (model.ContentGenerator[T], error)`

Both return a `model.ContentGenerator[T]`, which supports:

- `Generate(ctx context.Context) (T, model.GenerationMetadata, error)`
- `AddPromptContext(ctx context.Context, messageType model.ContextMessageType, content string)`
- `AddPromptContextProvider(ctx context.Context, provider model.PromptContextProvider)`

### Embedding Generators
Providers that support embeddings expose:

- `NewEmbeddingGenerator(opts ...model.GeneratorOption) (model.EmbeddingGenerator, error)`

Embedding usage is input-driven at call time:

- `Generate(ctx context.Context, input string) (model.EmbeddingVector, model.GenerationMetadata, error)`
- `GenerateBatch(ctx context.Context, inputs []string) (model.EmbeddingVectors, model.GenerationMetadata, error)`

### Audio Transcription Generators
Providers that support audio transcription expose:

- `NewAudioTranscriptionGenerator(filePath string, opts model.AudioOptions) (model.AudioTranscriptionGenerator, error)`

Audio usage:

- `Generate(ctx context.Context) (string, model.GenerationMetadata, error)`

`model.AudioOptions` notes:

- `Keywords []model.AudioKeyword` can be provided for domain-specific transcription hints.
- When `AudioOptions.Prompt` is empty, providers may add keyword hints as:
  - `Common missed words: <json-array-of-audio-keywords>`
- When `AudioOptions.Prompt` is set, that prompt is used as-is and keyword hints are not appended.

## Implemented LLM Providers
| Provider | Package | Content Generation (String + Structured) | Embeddings | Audio | Tools | MCP |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI | `pkg/llms/openai` | Yes | Yes | Yes | Yes | Native MCP (OpenAI Responses MCP tool) |
| Anthropic | `pkg/llms/anthropic` | Yes | No (returns unsupported in this library) | No | Yes | Native MCP (`mcp_servers`) |
| Bedrock | `pkg/llms/bedrock` | Yes | No | No | Yes | Tool-wrapped MCP (`pkg/mcp` adapter) |
| Gemini | `pkg/llms/gemini` | Yes | Yes | Yes | Yes | Tool-wrapped MCP (`pkg/mcp` adapter) |
| Ollama | `pkg/llms/ollama` | Yes | Yes | No | Yes | Tool-wrapped MCP (`pkg/mcp` adapter) |

Notes:
- OpenAI content generation (including tools and MCP) runs through the Responses API flow.
- Tool-wrapped MCP means MCP endpoints are bridged into regular tool calls via `pkg/mcp` so providers without native MCP can still use MCP tools.

## Tool Wrapped MCP
For providers that do not support MCP natively, this library uses a tool-wrapper approach to create the illusion of MCP support.

How it works:

1. You pass MCP server definitions with `model.WithMCPTools([]model.MCPTool{...})`.
2. The provider creates a `pkg/mcp.ToolAdapter` (an MCP client) per MCP server.
3. The adapter connects to the MCP server, initializes, and lists available tools.
4. MCP tools are converted into `model.Tool` definitions (`Name`, `Description`, `InputSchema`, `Handler`).
5. The provider gives those wrapped tools to the model as normal function tools.
6. When the model calls a tool, the adapter executes the MCP `CallTool` request and returns normalized output.

This gives non-native providers a consistent MCP experience without requiring provider-native MCP APIs.

# License
This is licensed under Apache 2.0, so feel free to use it in your projects and contribute to it as well.

## Disclaimer
This software is provided "AS IS", without warranty of any kind.
Use at your own risk.
