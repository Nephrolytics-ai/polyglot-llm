# Tests Directory

This directory is for tests that require external dependencies.

Examples include:
- Live network calls
- Third-party APIs (for example, OpenAI)
- External services that are not fully mocked

Tests here should remain deterministic where possible and skip clearly when required environment variables or credentials are missing.

Current credential variable conventions:
- `RUN_MCP_TEST=true` to enable MCP integration tests (one test per provider implementation)
- `MCP_SERVER_URL` for MCP integration tests server URL
- `MCP_SERVER_AUTHORIZATION` for MCP integration tests `Authorization` header value
- `ANTHROPIC_API_KEY` for Anthropic-backed tests
- Optional Anthropic settings: `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`
- `OPEN_API_TOKEN` for OpenAI-backed tests
- Optional OpenAI audio setting: `OPENAI_AUDIO_MODEL` (defaults to `whisper-1`)
- `GEMINI_KEY` for Gemini-backed tests
- Optional Gemini audio setting: `GEMINI_AUDIO_MODEL` (defaults to `gemini-2.5-flash`)
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optional `AWS_REGION` for Bedrock-backed tests (or `AWS_PROFILE`)
- `RUN_OLLAMA_TESTS=true` to enable Ollama-backed tests (requires local Ollama instance and models)
- Optional Ollama settings: `OLLAMA_BASE_URL`, `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL`
- `HF_TOKEN` for HuggingFace-backed tests
- Optional HuggingFace settings: `HF_BASE_URL`, `HF_MODEL` (defaults to `Qwen/Qwen2.5-72B-Instruct`), `HF_EMBEDDING_MODEL` (defaults to `BAAI/bge-base-en-v1.5`)

MCP integration tests use:
- MCP server URL from `MCP_SERVER_URL`
- `Authorization` header from `MCP_SERVER_AUTHORIZATION`
