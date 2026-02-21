package tests

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/llms/anthropic"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type AnthropicIntegrationSuite struct {
	ExternalDependenciesSuite
	apiKey  string
	baseURL string
	model   string
}

type anthropicStructuredResponse struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

type anthropicToolStructuredResponse struct {
	Secret string `json:"secret"`
}

func (s *AnthropicIntegrationSuite) SetupSuite() {
	s.ExternalDependenciesSuite.SetupSuite()

	s.apiKey = strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY"))
	s.baseURL = strings.TrimSpace(os.Getenv("ANTHROPIC_BASE_URL"))
	s.model = strings.TrimSpace(os.Getenv("ANTHROPIC_MODEL"))
	if s.apiKey == "" {
		s.T().Skip("ANTHROPIC_API_KEY is not set; skipping external dependency integration test")
	}
	if s.model == "" {
		s.model = "claude-sonnet-4-6"
	}
}

func (s *AnthropicIntegrationSuite) generationOpts() []model.GeneratorOption {
	opts := []model.GeneratorOption{
		model.WithAuthToken(s.apiKey),
		model.WithModel(s.model),
		model.WithMaxTokens(256),
	}
	if s.baseURL != "" {
		opts = append(opts, model.WithURL(s.baseURL))
	}
	return opts
}

func (s *AnthropicIntegrationSuite) TestCreateGeneratorAndGenerate() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	generator, err := anthropic.NewStringContentGenerator(
		"Reply with one short sentence saying hello.",
		s.generationOpts()...,
	)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.NotEmpty(s.T(), strings.TrimSpace(output))
	assert.Equal(s.T(), "anthropic", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyModel])
}

func (s *AnthropicIntegrationSuite) TestCreateStructuredGeneratorAndGenerate() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	generator, err := anthropic.NewStructureContentGenerator[anthropicStructuredResponse](
		"Return JSON with fields status and message. Set status to ok and message to a short greeting.",
		s.generationOpts()...,
	)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.NotEmpty(s.T(), strings.TrimSpace(output.Status))
	assert.NotEmpty(s.T(), strings.TrimSpace(output.Message))
	assert.Equal(s.T(), "anthropic", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyModel])
}

func (s *AnthropicIntegrationSuite) TestCreateGeneratorAndGenerateWithTool() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	const toolSecret = "anthropic-tool-secret-123"
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

	opts := append([]model.GeneratorOption{}, s.generationOpts()...)
	opts = append(opts, model.WithTools(tools))

	generator, err := anthropic.NewStructureContentGenerator[anthropicToolStructuredResponse](
		"Call the get_secret_value tool and return JSON with only the field secret set to the exact tool value.",
		opts...,
	)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.GreaterOrEqual(s.T(), toolCalls.Load(), int32(1))
	assert.Equal(s.T(), toolSecret, strings.TrimSpace(output.Secret))
	assert.Equal(s.T(), "anthropic", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyModel])
}

func TestAnthropicIntegrationSuite(t *testing.T) {
	suite.Run(t, new(AnthropicIntegrationSuite))
}
