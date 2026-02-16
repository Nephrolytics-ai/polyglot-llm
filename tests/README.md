# Tests Directory

This directory is for tests that require external dependencies.

Examples include:
- Live network calls
- Third-party APIs (for example, OpenAI)
- External services that are not fully mocked

Tests here should remain deterministic where possible and skip clearly when required environment variables or credentials are missing.

Current credential variable conventions:
- `OPEN_API_TOKEN` for OpenAI-backed tests
- `GEMINI_KEY` for Gemini-backed tests
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optional `AWS_REGION` for Bedrock-backed tests (or `AWS_PROFILE`)
- `RUN_OLLAMA_TESTS=true` to enable Ollama-backed tests (requires local Ollama instance and models)
- Optional Ollama settings: `OLLAMA_BASE_URL`, `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL`
