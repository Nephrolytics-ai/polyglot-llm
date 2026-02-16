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
	maxToolRounds              = 12
)

type client struct {
	apiClient *ollamasdk.OllamaClient
	baseURL   string
}

func newClient(cfg model.GeneratorConfig) *client {
	baseURL := strings.TrimSpace(cfg.URL)
	if baseURL == "" {
		baseURL = strings.TrimSpace(os.Getenv("OLLAMA_BASE_URL"))
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	return &client{
		apiClient: ollamasdk.NewClient(baseURL),
		baseURL:   baseURL,
	}
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
