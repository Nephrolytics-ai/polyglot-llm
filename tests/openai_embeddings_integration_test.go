package tests

import (
	"context"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/llms/openai_response"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type OpenAIEmbeddingsIntegrationSuite struct {
	ExternalDependenciesSuite
	apiKey  string
	baseURL string
}

func (s *OpenAIEmbeddingsIntegrationSuite) SetupSuite() {
	s.ExternalDependenciesSuite.SetupSuite()

	s.apiKey = strings.TrimSpace(os.Getenv("OPEN_API_TOKEN"))
	s.baseURL = strings.TrimSpace(os.Getenv("OPENAI_BASE_URL"))
	if s.apiKey == "" {
		s.T().Skip("OPEN_API_TOKEN is not set; skipping external dependency integration test")
	}
}

func (s *OpenAIEmbeddingsIntegrationSuite) embeddingOpts() []model.GeneratorOption {
	opts := []model.GeneratorOption{
		model.WithAuthToken(s.apiKey),
		model.WithModel("text-embedding-3-small"),
	}
	if s.baseURL != "" {
		opts = append(opts, model.WithURL(s.baseURL))
	}
	return opts
}

func (s *OpenAIEmbeddingsIntegrationSuite) TestGenerateSingleEmbedding() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	generator, err := openai_response.NewEmbeddingGenerator(s.embeddingOpts()...)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	vector, metadata, err := generator.Generate(ctx, "Hello from integration test.")
	require.NoError(s.T(), err)
	require.NotEmpty(s.T(), vector)
	assert.Greater(s.T(), len(vector), 0)
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyModel])
	assert.Equal(s.T(), "1", metadata[model.MetadataKeyEmbeddingCount])
	assert.Equal(s.T(), strconv.Itoa(len(vector)), metadata[model.MetadataKeyEmbeddingDims])
}

func (s *OpenAIEmbeddingsIntegrationSuite) TestGenerateBatchEmbeddings() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	inputs := []string{
		"Kidney function and electrolyte balance.",
		"Glomerular filtration rate estimation details.",
	}

	generator, err := openai_response.NewEmbeddingGenerator(s.embeddingOpts()...)
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
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func TestOpenAIEmbeddingsIntegrationSuite(t *testing.T) {
	suite.Run(t, new(OpenAIEmbeddingsIntegrationSuite))
}
