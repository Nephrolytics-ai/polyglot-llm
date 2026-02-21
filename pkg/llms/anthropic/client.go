package anthropic

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
	providerName         = "anthropic"
	defaultModelName     = "claude-3-5-sonnet-latest"
	defaultBaseURL       = "https://api.anthropic.com"
	anthropicVersion     = "2023-06-01"
	anthropicMCPBeta     = "mcp-client-2025-11-20"
	defaultMaxTokens     = 1024
	maxToolRounds        = 12
	defaultHTTPTimeout   = 90 * time.Second
	envAnthropicAPIKey   = "ANTHROPIC_API_KEY"
	envAnthropicBaseURL  = "ANTHROPIC_BASE_URL"
	envAnthropicModel    = "ANTHROPIC_MODEL"
)

type apiClient struct {
	httpClient *http.Client
	baseURL    string
	apiKey     string
}

type flowUsageTotals struct {
	APICalls          int
	ToolRounds        int
	InputTokens       int64
	OutputTokens      int64
	TotalTokens       int64
	CachedInputTokens int64
	ReasoningTokens   int64
}

type anthropicUsage struct {
	InputTokens        int64 `json:"input_tokens"`
	OutputTokens       int64 `json:"output_tokens"`
	CacheReadInput     int64 `json:"cache_read_input_tokens"`
	CacheCreationInput int64 `json:"cache_creation_input_tokens"`
}

type anthropicContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

type anthropicMessage struct {
	Role    string                 `json:"role"`
	Content []anthropicContentBlock `json:"content"`
}

type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

type anthropicMCPToolConfiguration struct {
	Enabled      *bool    `json:"enabled,omitempty"`
	AllowedTools []string `json:"allowed_tools,omitempty"`
}

type anthropicMCPServer struct {
	Type               string                         `json:"type"`
	Name               string                         `json:"name"`
	URL                string                         `json:"url"`
	AuthorizationToken string                         `json:"authorization_token,omitempty"`
	ToolConfiguration  *anthropicMCPToolConfiguration `json:"tool_configuration,omitempty"`
}

type anthropicMessageRequest struct {
	Model       string               `json:"model"`
	MaxTokens   int                  `json:"max_tokens"`
	Temperature *float64             `json:"temperature,omitempty"`
	System      string               `json:"system,omitempty"`
	Messages    []anthropicMessage   `json:"messages"`
	Tools       []anthropicTool      `json:"tools,omitempty"`
	MCPServers  []anthropicMCPServer `json:"mcp_servers,omitempty"`
}

type anthropicMessageResponse struct {
	ID         string                 `json:"id"`
	Type       string                 `json:"type"`
	Role       string                 `json:"role"`
	Model      string                 `json:"model"`
	Content    []anthropicContentBlock `json:"content"`
	StopReason string                 `json:"stop_reason"`
	Usage      *anthropicUsage        `json:"usage"`
}

type anthropicErrorResponse struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

func newAPIClient(cfg model.GeneratorConfig) (*apiClient, error) {
	apiKey := strings.TrimSpace(cfg.AuthToken)
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv(envAnthropicAPIKey))
	}
	if apiKey == "" {
		return nil, utils.WrapIfNotNil(errors.New("auth token is required (set WithAuthToken or ANTHROPIC_API_KEY)"))
	}

	baseURL := strings.TrimSpace(cfg.URL)
	if baseURL == "" {
		baseURL = strings.TrimSpace(os.Getenv(envAnthropicBaseURL))
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

func (c *apiClient) createMessage(ctx context.Context, request anthropicMessageRequest, includeMCPBeta bool) (*anthropicMessageResponse, error) {
	requestBits, err := json.Marshal(request)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	httpRequest, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		c.baseURL+"/v1/messages",
		bytes.NewReader(requestBits),
	)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	httpRequest.Header.Set("content-type", "application/json")
	httpRequest.Header.Set("x-api-key", c.apiKey)
	httpRequest.Header.Set("anthropic-version", anthropicVersion)
	if includeMCPBeta {
		httpRequest.Header.Set("anthropic-beta", anthropicMCPBeta)
	}

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
		apiErr := anthropicErrorResponse{}
		message := strings.TrimSpace(string(responseBits))
		if unmarshalErr := json.Unmarshal(responseBits, &apiErr); unmarshalErr == nil {
			candidate := strings.TrimSpace(apiErr.Error.Message)
			if candidate != "" {
				message = candidate
			}
		}
		if message == "" {
			message = "unknown anthropic error"
		}
		return nil, utils.WrapIfNotNil(fmt.Errorf("anthropic API error (%d): %s", httpResponse.StatusCode, message))
	}

	response := anthropicMessageResponse{}
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

	fromEnv := strings.TrimSpace(os.Getenv(envAnthropicModel))
	if fromEnv != "" {
		return fromEnv
	}
	return defaultModelName
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

func accumulateUsageTotals(totals *flowUsageTotals, response *anthropicMessageResponse) {
	if totals == nil || response == nil {
		return
	}

	totals.APICalls++
	if response.Usage == nil {
		return
	}

	totals.InputTokens += response.Usage.InputTokens
	totals.OutputTokens += response.Usage.OutputTokens
	totals.TotalTokens += response.Usage.InputTokens + response.Usage.OutputTokens
	totals.CachedInputTokens += response.Usage.CacheReadInput + response.Usage.CacheCreationInput
}

func applyAnthropicMetadata(meta model.GenerationMetadata, response *anthropicMessageResponse, totals flowUsageTotals) {
	if meta == nil {
		return
	}

	meta[model.MetadataKeyAPICalls] = strconv.Itoa(totals.APICalls)
	meta[model.MetadataKeyToolRounds] = strconv.Itoa(totals.ToolRounds)
	meta[model.MetadataKeyInputTokens] = strconv.FormatInt(totals.InputTokens, 10)
	meta[model.MetadataKeyOutputTokens] = strconv.FormatInt(totals.OutputTokens, 10)
	meta[model.MetadataKeyTotalTokens] = strconv.FormatInt(totals.TotalTokens, 10)
	meta[model.MetadataKeyCachedInputTokens] = strconv.FormatInt(totals.CachedInputTokens, 10)
	meta[model.MetadataKeyReasoningTokens] = strconv.FormatInt(totals.ReasoningTokens, 10)

	if response == nil {
		return
	}
	if strings.TrimSpace(response.ID) != "" {
		meta[model.MetadataKeyResponseID] = response.ID
	}
	if strings.TrimSpace(response.StopReason) != "" {
		meta[model.MetadataKeyResponseStatus] = response.StopReason
	}
	if strings.TrimSpace(response.Model) != "" {
		meta[model.MetadataKeyModel] = response.Model
	}
}

func normalizeGeneratorOptionsForProvider(cfg model.GeneratorConfig, log logging.Logger) (model.GeneratorConfig, error) {
	if cfg.ReasoningLevel != nil {
		if cfg.IgnoreInvalidGeneratorOptions {
			if log != nil {
				log.Warnf("ignoring reasoning level for anthropic provider")
			}
			cfg.ReasoningLevel = nil
		} else {
			return cfg, utils.WrapIfNotNil(errors.New("reasoning level is not supported for anthropic provider"))
		}
	}
	return cfg, nil
}
