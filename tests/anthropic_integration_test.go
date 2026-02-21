package tests

import (
	"context"
	"os"
	"strings"
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

func (s *AnthropicIntegrationSuite) SetupSuite() {
	s.ExternalDependenciesSuite.SetupSuite()

	s.apiKey = strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY"))
	s.baseURL = strings.TrimSpace(os.Getenv("ANTHROPIC_BASE_URL"))
	s.model = strings.TrimSpace(os.Getenv("ANTHROPIC_MODEL"))
	if s.apiKey == "" {
		s.T().Skip("ANTHROPIC_API_KEY is not set; skipping external dependency integration test")
	}
	if s.model == "" {
		s.model = "claude-3-5-sonnet-latest"
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

func TestAnthropicIntegrationSuite(t *testing.T) {
	suite.Run(t, new(AnthropicIntegrationSuite))
}
