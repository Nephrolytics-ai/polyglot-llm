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

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/llms/ollama"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type OllamaIntegrationSuite struct {
	ExternalDependenciesSuite
	baseURL    string
	chatModel  string
	embedModel string
}

type ollamaStructuredResponse struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

type ollamaToolStructuredResponse struct {
	Secret string `json:"secret"`
}

func (s *OllamaIntegrationSuite) SetupSuite() {
	s.ExternalDependenciesSuite.SetupSuite()

	run, err := strconv.ParseBool(strings.TrimSpace(os.Getenv("RUN_OLLAMA_TESTS")))
	if err != nil || !run {
		s.T().Skip("RUN_OLLAMA_TESTS is not true; skipping Ollama integration tests")
	}

	s.baseURL = strings.TrimSpace(os.Getenv("OLLAMA_BASE_URL"))
	s.chatModel = strings.TrimSpace(os.Getenv("OLLAMA_CHAT_MODEL"))
	if s.chatModel == "" {
		s.chatModel = "gpt-oss:20b"
	}
	s.embedModel = strings.TrimSpace(os.Getenv("OLLAMA_EMBED_MODEL"))
	if s.embedModel == "" {
		s.embedModel = "nomic-embed-text"
	}
}

func (s *OllamaIntegrationSuite) generationOpts() []model.GeneratorOption {
	opts := []model.GeneratorOption{
		model.WithModel(s.chatModel),
	}
	if s.baseURL != "" {
		opts = append(opts, model.WithURL(s.baseURL))
	}
	return opts
}

func (s *OllamaIntegrationSuite) embeddingOpts() []model.GeneratorOption {
	opts := []model.GeneratorOption{
		model.WithModel(s.embedModel),
	}
	if s.baseURL != "" {
		opts = append(opts, model.WithURL(s.baseURL))
	}
	return opts
}

func (s *OllamaIntegrationSuite) TestStringGeneration() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	generator, err := ollama.NewStringContentGenerator("How are you today?", s.generationOpts()...)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.NotEmpty(s.T(), strings.TrimSpace(output))
	assert.Equal(s.T(), "ollama", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyModel])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func (s *OllamaIntegrationSuite) TestStructuredGeneration() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	generator, err := ollama.NewStructureContentGenerator[ollamaStructuredResponse](
		"Return JSON with fields status and message. Set status to ok and message to a short greeting.",
		s.generationOpts()...,
	)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.NotEmpty(s.T(), strings.TrimSpace(output.Status))
	assert.NotEmpty(s.T(), strings.TrimSpace(output.Message))
	assert.Equal(s.T(), "ollama", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func (s *OllamaIntegrationSuite) TestStructuredGenerationWithTool() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	const toolSecret = "ollama-tool-secret-123"
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

	generator, err := ollama.NewStructureContentGenerator[ollamaToolStructuredResponse](
		"Call get_secret_value and return JSON with only the field secret set to the exact tool value.",
		opts...,
	)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.GreaterOrEqual(s.T(), toolCalls.Load(), int32(1))
	assert.Equal(s.T(), toolSecret, strings.TrimSpace(output.Secret))
	assert.Equal(s.T(), "ollama", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func (s *OllamaIntegrationSuite) TestSingleEmbeddingGeneration() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	generator, err := ollama.NewEmbeddingGenerator(s.embeddingOpts()...)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	vector, metadata, err := generator.Generate(ctx, "Kidney function and electrolyte balance.")
	require.NoError(s.T(), err)
	require.NotEmpty(s.T(), vector)
	assert.Greater(s.T(), len(vector), 0)
	assert.Equal(s.T(), "1", metadata[model.MetadataKeyEmbeddingCount])
	assert.Equal(s.T(), strconv.Itoa(len(vector)), metadata[model.MetadataKeyEmbeddingDims])
	assert.Equal(s.T(), "ollama", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func (s *OllamaIntegrationSuite) TestBatchEmbeddingGeneration() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	inputs := []string{
		"Kidney function and electrolyte balance.",
		"Glomerular filtration rate estimation details.",
	}

	generator, err := ollama.NewEmbeddingGenerator(s.embeddingOpts()...)
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
	assert.Equal(s.T(), "ollama", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func TestOllamaIntegrationSuite(t *testing.T) {
	suite.Run(t, new(OllamaIntegrationSuite))
}
