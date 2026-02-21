# OpenAI Provider

This package provides the OpenAI implementation for the shared model abstractions.

Implementation note:
- Text and structured generation are implemented using the OpenAI **Responses API**.
- Tool calling and MCP tool integration in this package also run through the Responses API flow.
- Embeddings and audio transcription use the corresponding OpenAI endpoints in the Go SDK.
