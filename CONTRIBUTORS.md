# Contributors Guide

This guide documents the coding patterns used in this repository so new contributions stay consistent and reviewable.

## Repository layout

- `pkg/model`
  - Provider-agnostic contracts and options (`ContentGenerator`, `EmbeddingGenerator`, `AudioTranscriptionGenerator`, `GeneratorOption`, metadata keys).
- `pkg/llms/<provider>`
  - Provider implementations (`openai`, `anthropic`, `bedrock`, `gemini`, `ollama`).
  - Each provider should expose constructors that match the `pkg/model` function prototypes.
- `pkg/logging`
  - Shared logging abstraction (`Logger`) and factory (`NewLogger(ctx)` + optional custom factory).
- `pkg/utils/errorutils.go`
  - Shared error wrapping helpers. Use `WrapIfNotNil` for returned errors.
- `pkg/mcp`
  - MCP adapter and tool conversion helpers.
- `tests`
  - Integration/external-dependency suites (credential-gated, deterministic where possible).
- `tests/data`
  - Stable fixtures used by integration tests (for example, audio fixtures).

## Standard generator prototypes

Use the shared function shapes from `pkg/model` when adding or updating providers.

### String content generator

```go
// Creates a generator that returns plain text.
func NewStringContentGenerator(prompt string, opts ...model.GeneratorOption) (model.ContentGenerator[string], error)
```

Usage pattern:

```go
ctx := context.Background()

gen, err := openai.NewStringContentGenerator(
    "How are you today?",
    model.WithAuthToken(token),
    model.WithModel("gpt-5-mini"),
)
if err != nil {
    return err
}

text, meta, err := gen.Generate(ctx)
```

### Structured content generator

```go
// Creates a generator that returns a typed struct (JSON-backed output).
func NewStructureContentGenerator[T any](prompt string, opts ...model.GeneratorOption) (model.ContentGenerator[T], error)
```

Usage pattern:

```go
type response struct {
    Status  string `json:"status"`
    Message string `json:"message"`
}

gen, err := gemini.NewStructureContentGenerator[response](
    "Return JSON with fields status and message.",
    model.WithAuthToken(token),
)
if err != nil {
    return err
}

out, meta, err := gen.Generate(ctx)
```

### Embedding generator

```go
// Creates an embedding generator; input is passed to Generate/GenerateBatch.
func NewEmbeddingGenerator(opts ...model.GeneratorOption) (model.EmbeddingGenerator, error)
```

Usage pattern:

```go
gen, err := ollama.NewEmbeddingGenerator(model.WithModel("nomic-embed-text"))
if err != nil {
    return err
}

vec, meta, err := gen.Generate(ctx, "Kidney function and electrolyte balance.")
```

### Audio transcription generator

```go
// Creates an audio transcription generator from a file path and audio options.
func NewAudioTranscriptionGenerator(filePath string, opts model.AudioOptions) (model.AudioTranscriptionGenerator, error)
```

Usage pattern:

```go
gen, err := openai.NewAudioTranscriptionGenerator(
    "data/transcript_test1.m4a",
    model.AudioOptions{
        AuthToken: token,
        Model:     "whisper-1",
        Keywords: map[string]string{
            // key: canonical term, value: common misspellings/variants
            "creatinine": "creatnine,creatinin",
        },
    },
)
if err != nil {
    return err
}

transcript, meta, err := gen.Generate(ctx)
```

## Provider implementation pattern

When implementing a provider constructor or `Generate` method:

1. Validate required inputs early (for example, non-empty prompt or file path).
2. Resolve options into config (`model.ResolveGeneratorOpts(opts...)`) when using generator options.
3. Build provider client/config once in constructor.
4. In `Generate(ctx)`, create metadata and use `logging.NewLogger(ctx)` for logs.
5. Return normalized metadata keys from `model` where available.
6. Wrap all returned errors with `utils.WrapIfNotNil(err, "FunctionName")`.

## Logging conventions (`pkg/logging`)

- Do not print directly to stdout/stderr (`fmt.Print*`, `println`).
- Use `logging.NewLogger(ctx)` when logging is needed.
- Keep logs concise and useful:
  - `Debug/Debugf` for diagnostic details.
  - `Info/Infof` for high-level operation boundaries.
  - `Error/Errorf` on failure paths.
- Prefer structured, context-rich messages over noisy logs.

## Error handling conventions (`pkg/utils/errorutils.go`)

- Wrap returned errors with `errorutils.WrapIfNotNil`.
- Include function context string where useful:

```go
if err != nil {
    return nil, utils.WrapIfNotNil(err, "NewEmbeddingGenerator")
}
```

- Never panic/fatal for normal error flow in library code.
- If you need stack dumps for diagnosis, use `errorutils.PrintStack` with a logger.

## Integration test patterns (`tests`)

Integration tests in this repo follow a shared pattern:

- Base suite: embed `ExternalDependenciesSuite` from `tests/base_suite_test.go`.
  - Loads `SETTINGS_FILE` if set.
  - Otherwise attempts `$HOME/.env` (no failure if missing).
- Use `testify/suite` with `require` and `assert`.
- Skip clearly in `SetupSuite` when required env vars are missing.
- Use bounded contexts (`context.WithTimeout`, typically 120s or 180s).
- Prefer deterministic prompts/assertions:
  - Assert required fields/metadata.
  - For text outputs, assert stable substrings or non-empty normalized output.
  - For embeddings, assert count/dimension metadata and vector lengths.
- Keep external test credentials behind env vars documented in `tests/README.md`.

## Contributor checklist

Before opening a PR:

1. Match `pkg/model` prototypes for generator constructors.
2. Use `logging.NewLogger(ctx)` (no stdout/stderr printing).
3. Wrap returned errors with `utils.WrapIfNotNil`.
4. Keep changes scoped to the task; do not modify unrelated files.
5. Add/update tests using testify suite patterns.
6. For integration tests, gate on env vars and skip cleanly when unavailable.
7. Run `gofmt` on edited Go files.
