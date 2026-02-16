package tests

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"sync/atomic"
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

type bedrockToolStructuredResponse struct {
	Secret string `json:"secret"`
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

func (s *BedrockIntegrationSuite) TestStructuredGenerationWithTool() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	const toolSecret = "bedrock-tool-secret-123"
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

	generator, err := bedrock.NewStructureContentGenerator[bedrockToolStructuredResponse](
		"Call get_secret_value and return JSON with only the field secret set to the exact tool value.",
		opts...,
	)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.GreaterOrEqual(s.T(), toolCalls.Load(), int32(1))
	assert.Equal(s.T(), toolSecret, strings.TrimSpace(output.Secret))
	assert.Equal(s.T(), "bedrock", metadata[model.MetadataKeyProvider])
	assert.Equal(s.T(), bedrockTestModel, metadata[model.MetadataKeyModel])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
}

func TestBedrockIntegrationSuite(t *testing.T) {
	suite.Run(t, new(BedrockIntegrationSuite))
}
