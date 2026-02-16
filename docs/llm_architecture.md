# LLM Architecture Design

## Goals

- Provide a single abstraction for text and structured generation across providers.
- Keep provider integrations isolated in `pkg/llms/*`.
- Support local tools and MCP tools.
- Support prompt context enrichment (static context + runtime providers).
- Return standardized generation metadata from every provider.
- Avoid `panic`/`fatal`; all failures return wrapped errors.

## Package Layout

- `pkg/model/llm.go`
  - Core interfaces and shared types.
  - Option model (`GeneratorOption`) and resolved config (`GeneratorConfig`).
  - Tool and MCP tool models.
  - Metadata model (`GenerationMetadata`) and common keys.
- `pkg/llms/openai_response`
  - OpenAI Responses API implementation (current full implementation).
- `pkg/llms/anthopic`
  - Stub implementation, same interface and metadata contract.
- `pkg/llms/gemini`
  - Stub implementation, same interface and metadata contract.
- `pkg/logging`
  - Logging abstraction and factory support.
- `pkg/utils/errorutils.go`
  - Error wrapping and utility helpers.

## Core Abstraction

`ContentGenerator[T]` is the provider-agnostic interface:

- `Generate(ctx context.Context) (T, GenerationMetadata, error)`
- `AddPromptContext(ctx context.Context, messageType ContextMessageType, content string)`
- `AddPromptContextProvider(ctx context.Context, provider PromptContextProvider)`

### Prompt Context Model

- `PromptContext` has:
  - `MessageType` (`system`, `human`, `assistant`)
  - `Content`
- `PromptContextProvider` generates context at runtime:
  - `GenerateContext(ctx) ([]*PromptContext, error)`

At generation time, implementations merge:
- static contexts added via `AddPromptContext`
- dynamic contexts returned by registered providers

## Unified Options Model

One options set is used for both provider-level and generation-level settings:

- `WithAuthToken(string)`
- `WithURL(string)`
- `WithIgnoreInvalidGeneratorOptions(bool)`
- `WithModel(string)`
- `WithTemperature(float64)`
- `WithMaxTokens(int)`
- `WithReasoningLevel(ReasoningLevel)`
- `WithTools([]Tool)`
- `WithMCPTools([]MCPTool)`

## Metadata Contract

`GenerationMetadata` is `map[string]string`.

Common keys are defined in `pkg/model/llm.go`:

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

Providers can add more keys, but common keys should remain stable.

## OpenAI Responses Implementation

## Construction

- `NewStringContentGenerator(prompt, opts...)`
- `NewStructureContentGenerator[T](prompt, opts...)`

Both:
- validate prompt is non-empty
- resolve options into `GeneratorConfig`
- build an OpenAI client using `URL` and `AuthToken`

## Input Construction

OpenAI uses structured input items (`OfInputItemList`), not concatenated prompt strings.

`ContextMessageType` mapping:
- `System` -> `system`
- `Human` -> `user`
- `Assistant` -> `assistant`

The main prompt is always appended as a `user` message item.

## Tooling

### Local tools

`model.Tool` maps to OpenAI function tools:
- `Name`, optional `Description`
- JSON schema parameters from `InputSchema`
- local handler function invoked during tool rounds

### MCP tools

`model.MCPTool` maps to OpenAI MCP tools:
- `Name` -> `server_label`
- `URL` -> `server_url`
- `AllowedTools` -> `allowed_tools`
- `HTTPHeaders["Authorization"]` mapped to MCP authorization field
- remaining headers passed through as MCP headers

## Stateless Tool Loop (Zero Data Retention Compatible)

The OpenAI flow is intentionally stateless and does **not** use `previous_response_id`.

Execution loop:

1. Submit initial request with message input and tool definitions.
2. Convert response output items into reusable input items and append to local history.
3. If function calls exist:
   - execute local handlers
   - append `function_call_output` items to local history
   - resend request with full history as `OfInputItemList`
4. Repeat until no more function calls or max rounds reached.

This supports organizations with Zero Data Retention constraints.

## Reasoning + Option Validation

Model validation rules:
- If model is reasoning-capable and temperature is set:
  - error unless `IgnoreInvalidGeneratorOptions=true`, then drop temperature.
- If model is non-reasoning and reasoning effort is set:
  - error unless `IgnoreInvalidGeneratorOptions=true`, then drop reasoning effort.

Reasoning model heuristic currently treats these prefixes as reasoning:
- `o1`, `o3`, `o4`, `gpt-5`

For reasoning models, the request includes:
- `include: reasoning.encrypted_content`

This allows reasoning state to be carried in stateless chaining.

## Structured Output Path

For `NewStructureContentGenerator[T]`:

- Generates JSON schema for `T` via `invopop/jsonschema`.
- Sends schema through `responses.ResponseTextConfigParam` with strict mode.
- Parses `response.OutputText()` JSON into `T`.

## Metadata Population (OpenAI)

Each `Generate` call returns metadata containing:
- base keys (`provider`, `model`, `latency_ms`)
- aggregated usage over all API calls in the run:
  - input/output/total tokens
  - cached input tokens
  - reasoning tokens
  - API call count
  - tool round count
- final response identifiers:
  - response ID
  - response status

## Error Handling and Logging

- All errors are returned, never fatal/panic.
- Errors are wrapped with `utils.WrapIfNotNil(...)`.
- Logging goes through `pkg/logging.NewLogger(ctx)`.

## Known Tradeoffs / Future Work

- Stateless history can grow quickly (cost + context-window pressure).
- No automatic truncation/compaction strategy is implemented yet.
- Anthopic and Gemini providers currently keep interface compatibility but are stubs.
