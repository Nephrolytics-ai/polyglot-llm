package tests

import (
	"context"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/llms/bedrock"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/llms/gemini"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/llms/ollama"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/llms/openai_response"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

const (
	mcpPrompt     = "Using ptId 914 get the medical history for the patient."
	mcpNeedleText = "chronic kidney disease"
)

type MCPIntegrationSuite struct {
	ExternalDependenciesSuite

	openAIKey       string
	openAIBaseURL   string
	geminiKey       string
	geminiBaseURL   string
	ollamaBaseURL   string
	ollamaChatModel string
	mcpServerURL    string
	mcpAuthHeader   string
}

func (s *MCPIntegrationSuite) SetupSuite() {
	s.ExternalDependenciesSuite.SetupSuite()

	run, err := strconv.ParseBool(strings.TrimSpace(os.Getenv("RUN_MCP_TEST")))
	if err != nil || !run {
		s.T().Skip("RUN_MCP_TEST is not true; skipping MCP integration tests")
	}

	s.openAIKey = strings.TrimSpace(os.Getenv("OPEN_API_TOKEN"))
	s.openAIBaseURL = strings.TrimSpace(os.Getenv("OPENAI_BASE_URL"))
	s.geminiKey = strings.TrimSpace(os.Getenv("GEMINI_KEY"))
	s.geminiBaseURL = strings.TrimSpace(os.Getenv("GEMINI_BASE_URL"))
	s.ollamaBaseURL = strings.TrimSpace(os.Getenv("OLLAMA_BASE_URL"))
	s.ollamaChatModel = strings.TrimSpace(os.Getenv("OLLAMA_CHAT_MODEL"))
	s.mcpServerURL = strings.TrimSpace(os.Getenv("MCP_SERVER_URL"))
	s.mcpAuthHeader = strings.TrimSpace(os.Getenv("MCP_SERVER_AUTHORIZATION"))

	if s.mcpServerURL == "" {
		s.T().Skip("MCP_SERVER_URL is not set; skipping MCP integration tests")
	}
	if s.mcpAuthHeader == "" {
		s.T().Skip("MCP_SERVER_AUTHORIZATION is not set; skipping MCP integration tests")
	}

	if s.ollamaChatModel == "" {
		s.ollamaChatModel = "gpt-oss:20b"
	}
}

func (s *MCPIntegrationSuite) mcpOption() model.GeneratorOption {
	return model.WithMCPTools([]model.MCPTool{
		{
			Name:        "dev_lab_mcp",
			URL:         s.mcpServerURL,
			HTTPHeaders: map[string]string{"Authorization": s.mcpAuthHeader},
		},
	})
}

func (s *MCPIntegrationSuite) assertContainsNeedle(output string) {
	normalizedOutput := strings.ToLower(strings.TrimSpace(output))
	normalizedNeedle := strings.ToLower(strings.TrimSpace(mcpNeedleText))

	require.NotEmpty(s.T(), normalizedOutput)
	require.NotEmpty(s.T(), normalizedNeedle)
	assert.Contains(s.T(), normalizedOutput, normalizedNeedle)
}

func (s *MCPIntegrationSuite) TestOpenAIWithMCPTool() {
	if s.openAIKey == "" {
		s.T().Skip("OPEN_API_TOKEN is not set; skipping OpenAI MCP integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()

	opts := []model.GeneratorOption{
		model.WithAuthToken(s.openAIKey),
		model.WithModel("gpt-5-mini"),
		model.WithReasoningLevel(model.ReasoningLevelLow),
		model.WithMaxTokens(512),
		s.mcpOption(),
	}
	if s.openAIBaseURL != "" {
		opts = append(opts, model.WithURL(s.openAIBaseURL))
	}

	generator, err := openai_response.NewStringContentGenerator(mcpPrompt, opts...)
	require.NoError(s.T(), err)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	s.assertContainsNeedle(output)
	assert.Equal(s.T(), "openai_response", metadata[model.MetadataKeyProvider])
}

func (s *MCPIntegrationSuite) TestGeminiWithMCPTool() {
	if s.geminiKey == "" {
		s.T().Skip("GEMINI_KEY is not set; skipping Gemini MCP integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()

	opts := []model.GeneratorOption{
		model.WithAuthToken(s.geminiKey),
		model.WithModel("gemini-2.5-flash"),
		model.WithReasoningLevel(model.ReasoningLevelMed),
		model.WithMaxTokens(1024),
		s.mcpOption(),
	}
	if s.geminiBaseURL != "" {
		opts = append(opts, model.WithURL(s.geminiBaseURL))
	}

	generator, err := gemini.NewStringContentGenerator(mcpPrompt, opts...)
	require.NoError(s.T(), err)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	s.assertContainsNeedle(output)
	assert.Equal(s.T(), "gemini", metadata[model.MetadataKeyProvider])
}

func (s *MCPIntegrationSuite) TestBedrockWithMCPTool() {
	accessKeyID := strings.TrimSpace(os.Getenv("AWS_ACCESS_KEY_ID"))
	secretAccessKey := strings.TrimSpace(os.Getenv("AWS_SECRET_ACCESS_KEY"))
	profile := strings.TrimSpace(os.Getenv("AWS_PROFILE"))

	hasStaticKeys := accessKeyID != "" && secretAccessKey != ""
	hasProfile := profile != ""
	if !hasStaticKeys && !hasProfile {
		s.T().Skip("AWS credentials not configured; skipping Bedrock MCP integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()

	opts := []model.GeneratorOption{
		model.WithModel(bedrockTestModel),
		model.WithMaxTokens(512),
		model.WithTemperature(0.2),
		s.mcpOption(),
	}

	generator, err := bedrock.NewStringContentGenerator(mcpPrompt, opts...)
	require.NoError(s.T(), err)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	s.assertContainsNeedle(output)
	assert.Equal(s.T(), "bedrock", metadata[model.MetadataKeyProvider])
}

func (s *MCPIntegrationSuite) TestOllamaWithMCPTool() {
	runOllama, err := strconv.ParseBool(strings.TrimSpace(os.Getenv("RUN_OLLAMA_TESTS")))
	if err != nil || !runOllama {
		s.T().Skip("RUN_OLLAMA_TESTS is not true; skipping Ollama MCP integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()

	opts := []model.GeneratorOption{
		model.WithModel(s.ollamaChatModel),
		s.mcpOption(),
	}
	if s.ollamaBaseURL != "" {
		opts = append(opts, model.WithURL(s.ollamaBaseURL))
	}

	generator, err := ollama.NewStringContentGenerator(mcpPrompt, opts...)
	require.NoError(s.T(), err)

	output, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	s.assertContainsNeedle(output)
	assert.Equal(s.T(), "ollama", metadata[model.MetadataKeyProvider])
}

func TestMCPIntegrationSuite(t *testing.T) {
	suite.Run(t, new(MCPIntegrationSuite))
}
