package tests

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/llms/huggingface"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type HuggingFaceIntegrationSuite struct {
	ExternalDependenciesSuite
	apiKey         string
	model          string
	embeddingModel string
}

type huggingFaceStructuredResponse struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

type huggingFaceToolStructuredResponse struct {
	Secret string `json:"secret"`
}

func (s *HuggingFaceIntegrationSuite) SetupSuite() {
	s.ExternalDependenciesSuite.SetupSuite()

	s.apiKey = strings.TrimSpace(os.Getenv("HF_TOKEN"))
	if s.apiKey == "" {
		s.T().Skip("HF_TOKEN is not set; skipping external dependency integration test")
	}
	s.model = strings.TrimSpace(os.Getenv("HF_MODEL"))
	if s.model == "" {
		s.model = "Qwen/Qwen2.5-72B-Instruct"
	}
	s.embeddingModel = strings.TrimSpace(os.Getenv("HF_EMBEDDING_MODEL"))
	if s.embeddingModel == "" {
		s.embeddingModel = "BAAI/bge-base-en-v1.5"
	}
}

func (s *HuggingFaceIntegrationSuite) generationOpts() []model.GeneratorOption {
	return []model.GeneratorOption{
		model.WithAuthToken(s.apiKey),
		model.WithModel(s.model),
		model.WithMaxTokens(256),
	}
}

func (s *HuggingFaceIntegrationSuite) embeddingOpts() []model.GeneratorOption {
	return []model.GeneratorOption{
		model.WithAuthToken(s.apiKey),
		model.WithModel(s.embeddingModel),
	}
}

func (s *HuggingFaceIntegrationSuite) TestCreateGeneratorAndGenerate() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	generator, err := huggingface.NewStringContentGenerator(
		"Reply with one short sentence saying hello.",
		s.generationOpts()...,
	)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.NotEmpty(s.T(), strings.TrimSpace(output))
	assert.Equal(s.T(), "huggingface", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyModel])
}

func (s *HuggingFaceIntegrationSuite) TestCreateStructuredGeneratorAndGenerate() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	generator, err := huggingface.NewStructureContentGenerator[huggingFaceStructuredResponse](
		"Return JSON with fields status and message. Set status to ok and message to a short greeting.",
		s.generationOpts()...,
	)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.NotEmpty(s.T(), strings.TrimSpace(output.Status))
	assert.NotEmpty(s.T(), strings.TrimSpace(output.Message))
	assert.Equal(s.T(), "huggingface", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyModel])
}

func (s *HuggingFaceIntegrationSuite) TestCreateGeneratorAndGenerateWithTool() {
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()

	const toolSecret = "huggingface-tool-secret-123"
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

	generator, err := huggingface.NewStructureContentGenerator[huggingFaceToolStructuredResponse](
		"Call the get_secret_value tool and return JSON with only the field secret set to the exact tool value.",
		opts...,
	)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.GreaterOrEqual(s.T(), toolCalls.Load(), int32(1))
	assert.Equal(s.T(), toolSecret, strings.TrimSpace(output.Secret))
	assert.Equal(s.T(), "huggingface", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyModel])
}

func (s *HuggingFaceIntegrationSuite) TestGenerateSingleEmbedding() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	generator, err := huggingface.NewEmbeddingGenerator(s.embeddingOpts()...)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	vector, metadata, err := generator.Generate(ctx, "hello world")
	require.NoError(s.T(), err)
	assert.NotEmpty(s.T(), vector)
	assert.Greater(s.T(), len(vector), 0)
	assert.Equal(s.T(), "huggingface", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func (s *HuggingFaceIntegrationSuite) TestGenerateBatchEmbeddings() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	generator, err := huggingface.NewEmbeddingGenerator(s.embeddingOpts()...)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	vectors, metadata, err := generator.GenerateBatch(ctx, []string{"hello", "world"})
	require.NoError(s.T(), err)
	assert.Len(s.T(), vectors, 2)
	assert.Greater(s.T(), len(vectors[0]), 0)
	assert.Greater(s.T(), len(vectors[1]), 0)
	assert.Equal(s.T(), "huggingface", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyEmbeddingCount])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyEmbeddingDims])
}

func TestHuggingFaceIntegrationSuite(t *testing.T) {
	suite.Run(t, new(HuggingFaceIntegrationSuite))
}
