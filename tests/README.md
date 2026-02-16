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
