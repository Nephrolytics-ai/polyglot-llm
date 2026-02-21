package tests

import (
	"context"
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/llms/gemini"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type GeminiIntegrationSuite struct {
	ExternalDependenciesSuite
	apiKey  string
	baseURL string
}

type geminiStructuredResponse struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

type geminiToolStructuredResponse struct {
	Secret string `json:"secret"`
}

func (s *GeminiIntegrationSuite) SetupSuite() {
	s.ExternalDependenciesSuite.SetupSuite()

	s.apiKey = strings.TrimSpace(os.Getenv("GEMINI_KEY"))
	s.baseURL = strings.TrimSpace(os.Getenv("GEMINI_BASE_URL"))
	if s.apiKey == "" {
		s.T().Skip("GEMINI_KEY is not set; skipping external dependency integration test")
	}
}

func (s *GeminiIntegrationSuite) generationOpts() []model.GeneratorOption {
	opts := []model.GeneratorOption{
		model.WithAuthToken(s.apiKey),
		model.WithModel("gemini-2.5-flash"),
		model.WithMaxTokens(256),
		model.WithReasoningLevel(model.ReasoningLevelLow),
	}
	if s.baseURL != "" {
		opts = append(opts, model.WithURL(s.baseURL))
	}
	return opts
}

func (s *GeminiIntegrationSuite) embeddingOpts() []model.GeneratorOption {
	opts := []model.GeneratorOption{
		model.WithAuthToken(s.apiKey),
		model.WithModel("gemini-embedding-001"),
	}
	if s.baseURL != "" {
		opts = append(opts, model.WithURL(s.baseURL))
	}
	return opts
}

func (s *GeminiIntegrationSuite) TestStringGeneration() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	generator, err := gemini.NewStringContentGenerator("How are you today?", s.generationOpts()...)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.NotEmpty(s.T(), strings.TrimSpace(output))
	assert.Equal(s.T(), "gemini", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyModel])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func (s *GeminiIntegrationSuite) TestStructuredGeneration() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	generator, err := gemini.NewStructureContentGenerator[geminiStructuredResponse](
		"Return JSON with fields status and message. Set status to ok and message to a short greeting.",
		s.generationOpts()...,
	)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.NotEmpty(s.T(), strings.TrimSpace(output.Status))
	assert.NotEmpty(s.T(), strings.TrimSpace(output.Message))
	assert.Equal(s.T(), "gemini", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func (s *GeminiIntegrationSuite) TestStructuredGenerationWithTool() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	const toolSecret = "gemini-tool-secret-123"
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

	generator, err := gemini.NewStructureContentGenerator[geminiToolStructuredResponse](
		"Call get_secret_value and return JSON with only the field secret set to the exact tool value.",
		opts...,
	)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.GreaterOrEqual(s.T(), toolCalls.Load(), int32(1))
	assert.Equal(s.T(), toolSecret, strings.TrimSpace(output.Secret))
	assert.Equal(s.T(), "gemini", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func (s *GeminiIntegrationSuite) TestSingleEmbeddingGeneration() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	generator, err := gemini.NewEmbeddingGenerator(s.embeddingOpts()...)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	vector, metadata, err := generator.Generate(ctx, "Kidney function and electrolyte balance.")
	require.NoError(s.T(), err)
	require.NotEmpty(s.T(), vector)
	assert.Greater(s.T(), len(vector), 0)
	assert.Equal(s.T(), "1", metadata[model.MetadataKeyEmbeddingCount])
	assert.Equal(s.T(), strconv.Itoa(len(vector)), metadata[model.MetadataKeyEmbeddingDims])
	assert.Equal(s.T(), "gemini", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func (s *GeminiIntegrationSuite) TestBatchEmbeddingGeneration() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	inputs := []string{
		"Kidney function and electrolyte balance.",
		"Glomerular filtration rate estimation details.",
	}

	generator, err := gemini.NewEmbeddingGenerator(s.embeddingOpts()...)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	vectors, metadata, err := generator.GenerateBatch(ctx, inputs)
	require.NoError(s.T(), err)
	require.Len(s.T(), vectors, len(inputs))
	require.NotEmpty(s.T(), vectors[0])
	require.NotEmpty(s.T(), vectors[1])
	assert.Equal(s.T(), len(vectors[0]), len(vectors[1]))
	assert.Equal(s.T(), strconv.Itoa(len(inputs)), metadata[model.MetadataKeyEmbeddingCount])
	assert.Equal(s.T(), strconv.Itoa(len(vectors[0])), metadata[model.MetadataKeyEmbeddingDims])
	assert.Equal(s.T(), "gemini", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func TestGeminiIntegrationSuite(t *testing.T) {
	suite.Run(t, new(GeminiIntegrationSuite))
}
