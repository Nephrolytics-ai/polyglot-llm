package gemini

import (
	"context"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
	"google.golang.org/genai"
)

const (
	providerName               = "gemini"
	defaultGenerationModelName = "gemini-2.5-flash"
	defaultEmbeddingModelName  = "gemini-embedding-001"
	maxToolRounds              = 12
)

type generationTotals struct {
	APICalls        int
	ToolRounds      int
	InputTokens     int64
	OutputTokens    int64
	TotalTokens     int64
	CachedTokens    int64
	ReasoningTokens int64
}

func newAPIClient(ctx context.Context, cfg model.GeneratorConfig) (*genai.Client, error) {
	clientCfg := &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
	}

	token := strings.TrimSpace(cfg.AuthToken)
	if token == "" {
		token = strings.TrimSpace(os.Getenv("GEMINI_KEY"))
	}
	if token != "" {
		clientCfg.APIKey = token
	}

	baseURL := strings.TrimSpace(cfg.URL)
	if baseURL != "" {
		clientCfg.HTTPOptions = genai.HTTPOptions{
			BaseURL: baseURL,
		}
	}

	client, err := genai.NewClient(ctx, clientCfg)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}
	return client, nil
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

func resolveGenerationModelName(cfg model.GeneratorConfig) string {
	if cfg.Model != nil {
		name := strings.TrimSpace(*cfg.Model)
		if name != "" {
			return name
		}
	}
	return defaultGenerationModelName
}

func resolveEmbeddingModelName(cfg model.GeneratorConfig) string {
	if cfg.Model != nil {
		name := strings.TrimSpace(*cfg.Model)
		if name != "" {
			return name
		}
	}
	return defaultEmbeddingModelName
}

func accumulateGenerationTotals(totals *generationTotals, response *genai.GenerateContentResponse) {
	if totals == nil || response == nil || response.UsageMetadata == nil {
		return
	}

	usage := response.UsageMetadata
	totals.APICalls++
	totals.InputTokens += int64(usage.PromptTokenCount)
	totals.OutputTokens += int64(usage.CandidatesTokenCount)
	totals.TotalTokens += int64(usage.TotalTokenCount)
	totals.CachedTokens += int64(usage.CachedContentTokenCount)
	totals.ReasoningTokens += int64(usage.ThoughtsTokenCount)
}

func applyGenerateMetadata(meta model.GenerationMetadata, response *genai.GenerateContentResponse, totals generationTotals) {
	if meta == nil {
		return
	}

	meta[model.MetadataKeyAPICalls] = strconv.Itoa(totals.APICalls)
	meta[model.MetadataKeyToolRounds] = strconv.Itoa(totals.ToolRounds)
	meta[model.MetadataKeyInputTokens] = strconv.FormatInt(totals.InputTokens, 10)
	meta[model.MetadataKeyOutputTokens] = strconv.FormatInt(totals.OutputTokens, 10)
	meta[model.MetadataKeyTotalTokens] = strconv.FormatInt(totals.TotalTokens, 10)
	meta[model.MetadataKeyCachedInputTokens] = strconv.FormatInt(totals.CachedTokens, 10)
	meta[model.MetadataKeyReasoningTokens] = strconv.FormatInt(totals.ReasoningTokens, 10)

	if response == nil {
		return
	}
	if strings.TrimSpace(response.ResponseID) != "" {
		meta[model.MetadataKeyResponseID] = response.ResponseID
	}
	if len(response.Candidates) > 0 && response.Candidates[0] != nil {
		meta[model.MetadataKeyResponseStatus] = string(response.Candidates[0].FinishReason)
	}
}

func applyEmbeddingMetadata(meta model.GenerationMetadata, vectors model.EmbeddingVectors) {
	if meta == nil {
		return
	}

	meta[model.MetadataKeyEmbeddingCount] = strconv.Itoa(len(vectors))
	if len(vectors) > 0 {
		meta[model.MetadataKeyEmbeddingDims] = strconv.Itoa(len(vectors[0]))
	}
	meta[model.MetadataKeyOutputTokens] = "0"
}
