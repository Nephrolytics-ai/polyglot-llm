package tests

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/llms/openai_response"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type OpenAIResponsesIntegrationSuite struct {
	ExternalDependenciesSuite
	apiKey  string
	baseURL string
}

type basicStructuredResponse struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

type toolStructuredResponse struct {
	Secret string `json:"secret"`
}

func (s *OpenAIResponsesIntegrationSuite) SetupSuite() {
	s.ExternalDependenciesSuite.SetupSuite()

	s.apiKey = strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	if s.apiKey == "" {
		s.apiKey = strings.TrimSpace(os.Getenv("OPEN_API_TOKEN"))
	}
	s.baseURL = strings.TrimSpace(os.Getenv("OPENAI_BASE_URL"))
	if s.apiKey == "" {
		s.T().Skip("OPENAI_API_KEY/OPEN_API_TOKEN is not set; skipping external dependency integration test")
	}
}

func (s *OpenAIResponsesIntegrationSuite) TestCreateGeneratorAndGenerate() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	opts := []model.GeneratorOption{
		model.WithAuthToken(s.apiKey),
	}
	if s.baseURL != "" {
		opts = append(opts, model.WithURL(s.baseURL))
	}

	opts = append(opts,
		model.WithModel("gpt-5-mini"),
		model.WithReasoningLevel(model.ReasoningLevelLow),
		model.WithMaxTokens(256),
	)

	generator, err := openai_response.NewStringContentGenerator(
		"How are you today.",
		opts...,
	)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.NotEmpty(s.T(), strings.TrimSpace(output))
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func (s *OpenAIResponsesIntegrationSuite) TestCreateStructuredGeneratorAndGenerate() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	opts := []model.GeneratorOption{
		model.WithAuthToken(s.apiKey),
	}
	if s.baseURL != "" {
		opts = append(opts, model.WithURL(s.baseURL))
	}

	opts = append(opts,
		model.WithModel("gpt-5-mini"),
		model.WithReasoningLevel(model.ReasoningLevelLow),
		model.WithMaxTokens(256),
	)

	generator, err := openai_response.NewStructureContentGenerator[basicStructuredResponse](
		"Return JSON with fields status and message. Set status to ok and message to a short greeting.",
		opts...,
	)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.NotEmpty(s.T(), strings.TrimSpace(output.Status))
	assert.NotEmpty(s.T(), strings.TrimSpace(output.Message))
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func (s *OpenAIResponsesIntegrationSuite) TestCreateGeneratorAndGenerateWithTool() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	const toolSecret = "tool-secret-value-123"
	var toolCalls atomic.Int32

	tools := []model.Tool{
		{
			Name:        "get_secret_value",
			Description: "Returns a fixed secret value.",
			InputSchema: model.JSONSchema{
				"type":                 "object",
				"properties":           map[string]any{},
				"additionalProperties": false,
			},
			Handler: func(ctx context.Context, args json.RawMessage) (any, error) {
				toolCalls.Add(1)
				return map[string]any{
					"secret": toolSecret,
				}, nil
			},
		},
	}

	opts := []model.GeneratorOption{
		model.WithAuthToken(s.apiKey),
	}
	if s.baseURL != "" {
		opts = append(opts, model.WithURL(s.baseURL))
	}

	opts = append(opts,
		model.WithModel("gpt-5-mini"),
		model.WithReasoningLevel(model.ReasoningLevelLow),
		model.WithMaxTokens(256),
		model.WithTools(tools),
	)

	generator, err := openai_response.NewStructureContentGenerator[toolStructuredResponse](
		"Call the get_secret_value tool and return JSON with only the field secret set to the exact tool value.",
		opts...,
	)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.GreaterOrEqual(s.T(), toolCalls.Load(), int32(1))
	assert.Equal(s.T(), toolSecret, strings.TrimSpace(output.Secret))
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func TestOpenAIResponsesIntegrationSuite(t *testing.T) {
	suite.Run(t, new(OpenAIResponsesIntegrationSuite))
}
