# Tests Directory

This directory is for tests that require external dependencies.

Examples include:
- Live network calls
- Third-party APIs (for example, OpenAI)
- External services that are not fully mocked

Tests here should remain deterministic where possible and skip clearly when required environment variables or credentials are missing.

Current credential variable conventions:
- `RUN_MCP_TEST=true` to enable MCP integration tests (one test per provider implementation except the Anthopic stub)
- `MCP_SERVER_URL` for MCP integration tests server URL
- `MCP_SERVER_AUTHORIZATION` for MCP integration tests `Authorization` header value
- `OPEN_API_TOKEN` for OpenAI-backed tests
- `OPENAI_AUDIO_TEST_FILE` for OpenAI audio integration test input file path (for example, `.wav`, `.mp3`, `.webm`)
- Optional OpenAI audio setting: `OPENAI_AUDIO_MODEL` (defaults to `whisper-1`)
- `GEMINI_KEY` for Gemini-backed tests
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optional `AWS_REGION` for Bedrock-backed tests (or `AWS_PROFILE`)
- `RUN_OLLAMA_TESTS=true` to enable Ollama-backed tests (requires local Ollama instance and models)
- Optional Ollama settings: `OLLAMA_BASE_URL`, `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL`

MCP integration tests use:
- MCP server URL from `MCP_SERVER_URL`
- `Authorization` header from `MCP_SERVER_AUTHORIZATION`
