# LangChain Alignment Design for polyglot-llm

## Purpose

Define a design that keeps the current provider implementations in `pkg/llms/*` but adds the missing runtime abstractions that make LangChain Python productive at scale (composition, retries, callbacks, history, retrieval, and agent middleware).

This document is design-only and does not require code changes yet.

## Background

Current `polyglot-llm` already provides:

- provider-agnostic generation, embeddings, and audio interfaces (`pkg/model/llm.go`, `pkg/model/embedding.go`, `pkg/model/audio.go`)
- provider implementations for OpenAI, Anthropic, Gemini, Bedrock, Ollama, HuggingFace
- tool calling and MCP bridging (`pkg/mcp`)
- normalized metadata maps

Current gap versus LangChain is mostly the execution/runtime layer, not provider adapters.

## Design Goals

1. Keep existing provider factories and interfaces stable.
2. Add a composable runtime layer similar to LangChain `Runnable`.
3. Standardize observability with callback events and run context.
4. Add reusable abstractions for prompting, parsing, memory, retrieval, and storage.
5. Add agent orchestration primitives with middleware hooks.
6. Keep Go ergonomics: explicit context, strict typing, clear error surfaces.

## Non-Goals

1. Replacing existing provider packages.
2. Enforcing Python-style dynamic behavior at the cost of type safety.
3. Introducing external service dependencies as mandatory runtime components.

## Reference Concepts from LangChain

This design is modeled from these LangChain concepts:

- `Runnable`, `RunnableSequence`, `RunnableParallel`
- `RunnableRetry`, `RunnableWithFallbacks`, `RunnableWithMessageHistory`
- callback managers and event hooks
- output parsers
- prompt templates and message models
- retriever/vectorstore/store/cache abstractions
- agent factory + middleware hooks

## Proposed Architecture

### Layer 1: Keep Existing Provider Layer (No Breaking Changes)

Retain current contracts in `pkg/model/*` and current provider constructor APIs:

- `NewStringContentGenerator(...)`
- `NewStructureContentGenerator[T](...)`
- `NewEmbeddingGenerator(...)`
- `NewAudioTranscriptionGenerator(...)`

Rationale: this is already stable and used by tests and integrations.

### Layer 2: Add Runtime Execution Abstractions

New package: `pkg/runtime`

Core interfaces:

```go
type Runnable[In any, Out any] interface {
	Invoke(ctx context.Context, input In, cfg RunConfig) (Out, error)
	Batch(ctx context.Context, inputs []In, cfg RunConfig) ([]Out, error)
}
```

`RunConfig` (new):

```go
type RunConfig struct {
	RunID          string
	RunName        string
	Tags           []string
	Metadata       map[string]string
	MaxConcurrency int
	RecursionLimit int
	Callbacks      []CallbackHandler
	Configurable   map[string]any
}
```

Composition primitives:

- `Sequence[A,B,C]` style composition (`A -> B -> C`)
- `Parallel[In]` fan-out composition into keyed outputs
- `LambdaRunnable` wrapper for plain functions

Wrapper runnables:

- `WithRetry(...)`
- `WithFallbacks(...)`
- `WithMessageHistory(...)`

### Layer 3: Add Callback/Event System

New package: `pkg/callbacks`

Event lifecycle:

- `OnRunStart`
- `OnRunEnd`
- `OnRunError`
- `OnModelStart`
- `OnModelToken`
- `OnModelEnd`
- `OnToolStart`
- `OnToolEnd`
- `OnToolError`
- `OnRetrieverStart`
- `OnRetrieverEnd`
- `OnRetrieverError`

This is separate from `pkg/logging`:

- `logging` remains logger-focused
- `callbacks` is execution-trace focused

### Layer 4: Add Prompt and Message Abstractions

New packages:

- `pkg/messages`
- `pkg/prompts`

`messages` should support:

- role enum (`system`, `user`, `assistant`, `tool`)
- text and future multimodal blocks
- provider-neutral message model

`prompts` should support:

- templated prompt rendering with required/optional vars
- prompt invocation as a runnable (`map[string]any -> PromptValue`)
- optional parser attachment

### Layer 5: Add Output Parsers

New package: `pkg/parsers`

Core parser interface:

```go
type OutputParser[T any] interface {
	Parse(ctx context.Context, text string) (T, error)
}
```

Initial parsers:

- string passthrough parser
- strict JSON parser to `map[string]any`
- typed JSON parser to `T` (schema-guided optional)

### Layer 6: Add Memory and Chat History

New package: `pkg/memory`

Core history interface:

```go
type ChatHistory interface {
	GetMessages(ctx context.Context) ([]messages.Message, error)
	AddMessages(ctx context.Context, msgs []messages.Message) error
	Clear(ctx context.Context) error
}
```

Initial implementation:

- in-memory history

Then wrapper:

- `runtime.WithMessageHistory(runnable, historyFactory, options...)`

### Layer 7: Add Retrieval and Storage Abstractions

New packages:

- `pkg/documents`
- `pkg/retrieval`
- `pkg/vectorstore`
- `pkg/store`
- `pkg/cache`

Core interfaces:

- `Document` (id, content, metadata)
- `Retriever` (`query -> []Document`)
- `VectorStore` (add/search/delete/get)
- `KVStore` batch interface (`mget`, `mset`, `mdelete`, `yield_keys`)
- `LLMCache` (`lookup/update/clear`)

Initial in-memory implementations first, provider-backed implementations later.

### Layer 8: Add Agent Runtime and Middleware

New package: `pkg/agent`

Core agent loop:

1. run model
2. inspect tool calls
3. run tools
4. append tool outputs
5. repeat until terminal response or limit

Middleware hooks:

- `BeforeModel`
- `AfterModel`
- `WrapModelCall`
- `WrapToolCall`

This makes policy and control logic pluggable (retry limits, filtering, human gate, etc.).

## Integration with Existing Types

### Keep Existing `ContentGenerator` and Add Adapters

Add adapter helpers that wrap existing generators as runnables:

- `RunnableFromStringGenerator(generator ContentGenerator[string])`
- `RunnableFromStructuredGenerator[T](generator ContentGenerator[T])`

This preserves all current provider logic and lets runtime abstractions be adopted incrementally.

### Metadata Strategy

Keep existing `GenerationMetadata map[string]string` for provider returns.

In runtime, include:

- run metadata in `RunConfig.Metadata`
- provider metadata from generator result
- callback events carrying both

No metadata schema break required.

## Error Handling Strategy

1. Preserve existing provider behavior:
   - strict option validation by default
   - opt-in ignore with `WithIgnoreInvalidGeneratorOptions`
2. Runtime wrappers should return wrapped errors with operation context.
3. Retries/fallbacks should expose final and intermediate failures via callbacks.

## Concurrency Model

1. `Batch` supports bounded concurrency via `RunConfig.MaxConcurrency`.
2. Parallel runnable uses independent child runs for each branch.
3. History and callback managers must be thread-safe.

## Versioning and Compatibility

This design is additive:

1. Existing APIs remain as-is.
2. New packages are optional adoption paths.
3. No immediate deprecations required.

Future deprecation candidates (later, optional):

- duplicate/parallel abstractions that become redundant after runtime is stable.

## Testing Plan

### Contract Tests (New)

Add reusable contract-style tests for:

- runtime invoke/batch/parallel semantics
- retry/fallback behaviors
- callback event ordering and payloads
- prompt rendering and parser failures
- history wrapper persistence behavior
- vectorstore/retriever/store/cache contracts

### Provider Compatibility Tests (Existing + Extended)

Keep current tests in `tests/*_integration_test.go`.

Add compatibility tests that verify provider generators can be wrapped in runtime adapters without behavior regressions.

## Implementation Phases

### Phase 1 (Runtime Foundation)

- `pkg/runtime` (Runnable, Sequence, Parallel, RunConfig)
- `pkg/callbacks` (basic handlers + manager)
- generator-to-runnable adapters

### Phase 2 (Resilience + History)

- retry wrapper
- fallback wrapper
- chat history interface + in-memory implementation
- message history runnable wrapper

### Phase 3 (Prompt + Parsers)

- prompt template abstraction
- output parsers
- parser wrappers for runnable output

### Phase 4 (Retrieval + Storage)

- document/retriever/vectorstore/store/cache interfaces
- in-memory implementations

### Phase 5 (Agent Middleware)

- agent loop package
- middleware hook framework
- standard policy middleware examples

## Risks and Mitigations

1. Risk: abstraction overlap with existing generators.
   Mitigation: keep runtime as adapter layer first, no forced migration.

2. Risk: callback complexity.
   Mitigation: start with a minimal event set and strict payload types.

3. Risk: over-generalization.
   Mitigation: phase-gate features and prove with contract tests before expansion.

4. Risk: behavior drift across providers.
   Mitigation: require provider compatibility tests through adapters.

## Acceptance Criteria

1. Existing provider integration tests continue passing without API changes.
2. A runnable sequence can compose prompt -> model -> parser in a provider-neutral way.
3. Retries and fallbacks work uniformly across at least OpenAI, Gemini, and Anthropic.
4. Callback traces include model and tool lifecycle events with run IDs.
5. In-memory history, vectorstore, and cache implementations pass contract tests.

## Open Questions

1. Should streaming be part of Phase 1 (`Stream`/`AStream`) or deferred to Phase 2?
2. Should callback payload metadata be `map[string]string` only, or allow `map[string]any`?
3. Do we want one unified `Message` model now, or keep provider-specific context mapping until Phase 3?
4. Should agent middleware ship in core package or under `pkg/agent/experimental` first?
