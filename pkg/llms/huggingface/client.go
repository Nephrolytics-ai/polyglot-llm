package huggingface

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/logging"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
)

const (
	providerName              = "huggingface"
	defaultModelName          = "Qwen/Qwen2.5-72B-Instruct"
	defaultEmbeddingModelName = "BAAI/bge-base-en-v1.5"
	defaultBaseURL            = "https://router.huggingface.co"
	defaultMaxTokens          = 1024
	maxToolRounds             = 12
	defaultHTTPTimeout        = 90 * time.Second
	envHFToken                = "HF_TOKEN"
	envHFBaseURL              = "HF_BASE_URL"
	envHFModel                = "HF_MODEL"
)

type apiClient struct {
	httpClient *http.Client
	baseURL    string
	apiKey     string
}

type flowUsageTotals struct {
	APICalls     int
	ToolRounds   int
	InputTokens  int64
	OutputTokens int64
	TotalTokens  int64
}

type chatMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content,omitempty"`
	ToolCalls  []chatToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type chatToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function chatFunctionCall `json:"function"`
}

type chatFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatTool struct {
	Type     string       `json:"type"`
	Function chatFunction `json:"function"`
}

type chatFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type chatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	Tools       []chatTool    `json:"tools,omitempty"`
}

type chatCompletionResponse struct {
	ID      string                 `json:"id"`
	Model   string                 `json:"model"`
	Choices []chatCompletionChoice `json:"choices"`
	Usage   *chatCompletionUsage   `json:"usage"`
}

type chatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type chatCompletionUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

type chatCompletionErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func newAPIClient(cfg model.GeneratorConfig) (*apiClient, error) {
	apiKey := strings.TrimSpace(cfg.AuthToken)
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv(envHFToken))
	}
	if apiKey == "" {
		return nil, utils.WrapIfNotNil(errors.New("auth token is required (set WithAuthToken or HF_TOKEN)"))
	}

	baseURL := strings.TrimSpace(cfg.URL)
	if baseURL == "" {
		baseURL = strings.TrimSpace(os.Getenv(envHFBaseURL))
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	baseURL = strings.TrimSuffix(baseURL, "/")

	return &apiClient{
		httpClient: &http.Client{Timeout: defaultHTTPTimeout},
		baseURL:    baseURL,
		apiKey:     apiKey,
	}, nil
}

func (c *apiClient) createChatCompletion(ctx context.Context, request chatCompletionRequest) (*chatCompletionResponse, error) {
	requestBits, err := json.Marshal(request)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	httpRequest, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		c.baseURL+"/v1/chat/completions",
		bytes.NewReader(requestBits),
	)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Authorization", "Bearer "+c.apiKey)

	httpResponse, err := c.httpClient.Do(httpRequest)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}
	defer httpResponse.Body.Close()

	responseBits, err := io.ReadAll(httpResponse.Body)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	if httpResponse.StatusCode < 200 || httpResponse.StatusCode >= 300 {
		apiErr := chatCompletionErrorResponse{}
		message := strings.TrimSpace(string(responseBits))
		if unmarshalErr := json.Unmarshal(responseBits, &apiErr); unmarshalErr == nil {
			candidate := strings.TrimSpace(apiErr.Error.Message)
			if candidate != "" {
				message = candidate
			}
		}
		if message == "" {
			message = "unknown huggingface error"
		}
		return nil, utils.WrapIfNotNil(fmt.Errorf("huggingface API error (%d): %s", httpResponse.StatusCode, message))
	}

	response := chatCompletionResponse{}
	err = json.Unmarshal(responseBits, &response)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}
	return &response, nil
}

func resolveModelName(cfg model.GeneratorConfig) string {
	if cfg.Model != nil {
		name := strings.TrimSpace(*cfg.Model)
		if name != "" {
			return name
		}
	}

	fromEnv := strings.TrimSpace(os.Getenv(envHFModel))
	if fromEnv != "" {
		return fromEnv
	}
	return defaultModelName
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

func resolveMaxTokens(cfg model.GeneratorConfig) int {
	if cfg.MaxTokens != nil && *cfg.MaxTokens > 0 {
		return *cfg.MaxTokens
	}
	return defaultMaxTokens
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

func accumulateUsageTotals(totals *flowUsageTotals, response *chatCompletionResponse) {
	if totals == nil || response == nil {
		return
	}

	totals.APICalls++
	if response.Usage == nil {
		return
	}

	totals.InputTokens += response.Usage.PromptTokens
	totals.OutputTokens += response.Usage.CompletionTokens
	totals.TotalTokens += response.Usage.TotalTokens
}

func applyHuggingFaceMetadata(meta model.GenerationMetadata, response *chatCompletionResponse, totals flowUsageTotals) {
	if meta == nil {
		return
	}

	meta[model.MetadataKeyAPICalls] = strconv.Itoa(totals.APICalls)
	meta[model.MetadataKeyToolRounds] = strconv.Itoa(totals.ToolRounds)
	meta[model.MetadataKeyInputTokens] = strconv.FormatInt(totals.InputTokens, 10)
	meta[model.MetadataKeyOutputTokens] = strconv.FormatInt(totals.OutputTokens, 10)
	meta[model.MetadataKeyTotalTokens] = strconv.FormatInt(totals.TotalTokens, 10)
	meta[model.MetadataKeyCachedInputTokens] = "0"
	meta[model.MetadataKeyReasoningTokens] = "0"

	if response == nil {
		return
	}
	if strings.TrimSpace(response.ID) != "" {
		meta[model.MetadataKeyResponseID] = response.ID
	}
	if len(response.Choices) > 0 && strings.TrimSpace(response.Choices[0].FinishReason) != "" {
		meta[model.MetadataKeyResponseStatus] = response.Choices[0].FinishReason
	}
	if strings.TrimSpace(response.Model) != "" {
		meta[model.MetadataKeyModel] = response.Model
	}
}

func normalizeGeneratorOptionsForProvider(cfg model.GeneratorConfig, log logging.Logger) (model.GeneratorConfig, error) {
	if cfg.ReasoningLevel != nil {
		if cfg.IgnoreInvalidGeneratorOptions {
			if log != nil {
				log.Warnf("ignoring reasoning level for huggingface provider")
			}
			cfg.ReasoningLevel = nil
		} else {
			return cfg, utils.WrapIfNotNil(errors.New("reasoning level is not supported for huggingface provider"))
		}
	}
	return cfg, nil
}
