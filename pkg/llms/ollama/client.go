package ollama

import (
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	ollamasdk "github.com/rozoomcool/go-ollama-sdk"
)

const (
	providerName               = "ollama"
	defaultGenerationModelName = "llama3.1"
	defaultEmbeddingModelName  = "nomic-embed-text"
	defaultBaseURL             = "http://localhost:11434"
	envOllamaAuthToken         = "OLLAMA_AUTH_TOKEN"
	maxToolRounds              = 12
)

type client struct {
	apiClient *ollamasdk.OllamaClient
	baseURL   string
	authToken string
}

func newClient(cfg model.GeneratorConfig) *client {
	baseURL := strings.TrimSpace(cfg.URL)
	if baseURL == "" {
		baseURL = strings.TrimSpace(os.Getenv("OLLAMA_BASE_URL"))
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	authToken := strings.TrimSpace(cfg.AuthToken)
	if authToken == "" {
		authToken = strings.TrimSpace(os.Getenv(envOllamaAuthToken))
	}

	return &client{
		apiClient: ollamasdk.NewClient(baseURL),
		baseURL:   baseURL,
		authToken: authToken,
	}
}

func (c *client) applyAuthHeader(headers interface{ Set(string, string) }) {
	if c == nil || headers == nil {
		return
	}

	token := strings.TrimSpace(c.authToken)
	if token == "" {
		return
	}

	if strings.HasPrefix(strings.ToLower(token), "bearer ") {
		headers.Set("Authorization", token)
		return
	}

	headers.Set("Authorization", "Bearer "+token)
}

func resolveGenerationModelName(cfg model.GeneratorConfig) string {
	if cfg.Model != nil {
		modelName := strings.TrimSpace(*cfg.Model)
		if modelName != "" {
			return modelName
		}
	}
	return defaultGenerationModelName
}

func resolveEmbeddingModelName(cfg model.GeneratorConfig) string {
	if cfg.Model != nil {
		modelName := strings.TrimSpace(*cfg.Model)
		if modelName != "" {
			return modelName
		}
	}
	return defaultEmbeddingModelName
}

func initMetadata(modelName string) model.GenerationMetadata {
	if strings.TrimSpace(modelName) == "" {
		modelName = "unknown"
	}

	return model.GenerationMetadata{
		model.MetadataKeyProvider: providerName,
		model.MetadataKeyModel:    modelName,
	}
}

func setLatencyMetadata(meta model.GenerationMetadata, start time.Time) {
	if meta == nil {
		return
	}
	meta[model.MetadataKeyLatencyMs] = strconv.FormatInt(time.Since(start).Milliseconds(), 10)
}
