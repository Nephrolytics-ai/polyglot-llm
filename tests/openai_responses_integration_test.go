package tests

import (
	"context"
	"os"
	"strings"
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

func (s *OpenAIResponsesIntegrationSuite) SetupSuite() {
	s.ExternalDependenciesSuite.SetupSuite()

	s.apiKey = strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	s.baseURL = strings.TrimSpace(os.Getenv("OPENAI_BASE_URL"))
	if s.apiKey == "" {
		s.T().Skip("OPENAI_API_KEY is not set; skipping external dependency integration test")
	}
}

func (s *OpenAIResponsesIntegrationSuite) TestCreateGeneratorAndGenerate() {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	llmOpts := []model.LLMOption{
		model.WithAuthToken(s.apiKey),
	}
	if s.baseURL != "" {
		llmOpts = append(llmOpts, model.WithURL(s.baseURL))
	}

	generator, err := openai_response.NewStringContentGenerator(
		"How are you today.",
		llmOpts,
		model.WithModel("gpt-5-mini"),
		model.WithReasoningLevel(model.ReasoningLevelLow),
		model.WithMaxTokens(256),
	)
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	output, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.NotEmpty(s.T(), strings.TrimSpace(output))
}

func TestOpenAIResponsesIntegrationSuite(t *testing.T) {
	suite.Run(t, new(OpenAIResponsesIntegrationSuite))
}
