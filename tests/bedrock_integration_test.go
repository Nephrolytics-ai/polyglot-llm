package tests

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/llms/bedrock"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

const bedrockTestModel = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

type BedrockIntegrationSuite struct {
	ExternalDependenciesSuite
}

type bedrockStructuredResponse struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

func (s *BedrockIntegrationSuite) SetupSuite() {
	s.ExternalDependenciesSuite.SetupSuite()

	accessKeyID := strings.TrimSpace(os.Getenv("AWS_ACCESS_KEY_ID"))
	secretAccessKey := strings.TrimSpace(os.Getenv("AWS_SECRET_ACCESS_KEY"))
	profile := strings.TrimSpace(os.Getenv("AWS_PROFILE"))

	hasStaticKeys := accessKeyID != "" && secretAccessKey != ""
	hasProfile := profile != ""
	if !hasStaticKeys && !hasProfile {
		s.T().Skip("AWS credentials not configured; set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE")
	}
}

func (s *BedrockIntegrationSuite) generationOpts() []model.GeneratorOption {
	return []model.GeneratorOption{
		model.WithModel(bedrockTestModel),
		model.WithMaxTokens(256),
		model.WithTemperature(0.2),
	}
}

func (s *BedrockIntegrationSuite) TestStringGeneration() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	generator, err := bedrock.NewStringContentGenerator(
		"How are you today?",
		s.generationOpts()...,
	)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.NotEmpty(s.T(), strings.TrimSpace(output))
	assert.Equal(s.T(), "bedrock", metadata[model.MetadataKeyProvider])
	assert.Equal(s.T(), bedrockTestModel, metadata[model.MetadataKeyModel])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func (s *BedrockIntegrationSuite) TestStructuredGeneration() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	generator, err := bedrock.NewStructureContentGenerator[bedrockStructuredResponse](
		"Return JSON with fields status and message. Set status to ok and message to a short greeting.",
		s.generationOpts()...,
	)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.NotEmpty(s.T(), strings.TrimSpace(output.Status))
	assert.NotEmpty(s.T(), strings.TrimSpace(output.Message))
	assert.Equal(s.T(), "bedrock", metadata[model.MetadataKeyProvider])
	assert.Equal(s.T(), bedrockTestModel, metadata[model.MetadataKeyModel])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func TestBedrockIntegrationSuite(t *testing.T) {
	suite.Run(t, new(BedrockIntegrationSuite))
}
