package tests

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/llms/gemini"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/llms/openai"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type OpenAIAudioIntegrationSuite struct {
	ExternalDependenciesSuite
	apiKey    string
	baseURL   string
	modelName string
}

type GeminiAudioIntegrationSuite struct {
	ExternalDependenciesSuite
	apiKey    string
	baseURL   string
	modelName string
}

const audioFixturePath = "data/transcript_test1.m4a"

func (s *OpenAIAudioIntegrationSuite) SetupSuite() {
	s.ExternalDependenciesSuite.SetupSuite()

	s.apiKey = strings.TrimSpace(os.Getenv("OPEN_API_TOKEN"))
	s.baseURL = strings.TrimSpace(os.Getenv("OPENAI_BASE_URL"))
	s.modelName = strings.TrimSpace(os.Getenv("OPENAI_AUDIO_MODEL"))

	if s.apiKey == "" {
		s.T().Skip("OPEN_API_TOKEN is not set; skipping external dependency integration test")
	}
	if _, err := os.Stat(audioFixturePath); err != nil {
		s.T().Skipf("%s is not accessible (%v); skipping OpenAI audio integration test", audioFixturePath, err)
	}
	if s.modelName == "" {
		s.modelName = "whisper-1"
	}
}

func (s *OpenAIAudioIntegrationSuite) audioOptions() model.AudioOptions {
	return model.AudioOptions{
		AuthToken: s.apiKey,
		URL:       s.baseURL,
		Model:     s.modelName,
		Keywords: map[string]string{
			"afib":       "atrial fibrillation",
			"creatinine": "creatinine",
		},
	}
}

func (s *OpenAIAudioIntegrationSuite) TestCreateGeneratorAndGenerateTranscript() {
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()

	generator, err := openai.NewAudioTranscriptionGenerator(audioFixturePath, s.audioOptions())
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	transcript, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.NotEmpty(s.T(), strings.TrimSpace(transcript))
	normalizedTranscript := strings.ToLower(transcript)
	assert.Contains(s.T(), normalizedTranscript, strings.ToLower("egfr"))
	assert.Equal(s.T(), "openai", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyModel])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyInputTokens])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyOutputTokens])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyTotalTokens])
}

func TestOpenAIAudioIntegrationSuite(t *testing.T) {
	suite.Run(t, new(OpenAIAudioIntegrationSuite))
}

func (s *GeminiAudioIntegrationSuite) SetupSuite() {
	s.ExternalDependenciesSuite.SetupSuite()

	s.apiKey = strings.TrimSpace(os.Getenv("GEMINI_KEY"))
	s.baseURL = strings.TrimSpace(os.Getenv("GEMINI_BASE_URL"))
	s.modelName = strings.TrimSpace(os.Getenv("GEMINI_AUDIO_MODEL"))

	if s.apiKey == "" {
		s.T().Skip("GEMINI_KEY is not set; skipping external dependency integration test")
	}
	if _, err := os.Stat(audioFixturePath); err != nil {
		s.T().Skipf("%s is not accessible (%v); skipping Gemini audio integration test", audioFixturePath, err)
	}
	if s.modelName == "" {
		s.modelName = "gemini-2.5-flash"
	}
}

func (s *GeminiAudioIntegrationSuite) audioOptions() model.AudioOptions {
	return model.AudioOptions{
		AuthToken: s.apiKey,
		URL:       s.baseURL,
		Model:     s.modelName,
		Keywords: map[string]string{
			"egfr":       "estimated glomerular filtration rate",
			"creatinine": "creatinine",
		},
	}
}

func (s *GeminiAudioIntegrationSuite) TestCreateGeneratorAndGenerateTranscript() {
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()

	generator, err := gemini.NewAudioTranscriptionGenerator(audioFixturePath, s.audioOptions())
	require.NoError(s.T(), err)
	require.NotNil(s.T(), generator)

	transcript, metadata, err := generator.Generate(ctx)
	require.NoError(s.T(), err)
	assert.NotEmpty(s.T(), strings.TrimSpace(transcript))
	normalizedTranscript := strings.ToLower(transcript)
	assert.Contains(s.T(), normalizedTranscript, strings.ToLower("egfr"))
	assert.Equal(s.T(), "gemini", metadata[model.MetadataKeyProvider])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyLatencyMs])
	assert.NotEmpty(s.T(), metadata[model.MetadataKeyModel])
}

func TestGeminiAudioIntegrationSuite(t *testing.T) {
	suite.Run(t, new(GeminiAudioIntegrationSuite))
}
