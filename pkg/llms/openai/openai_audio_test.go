package openai

import (
	"context"
	"testing"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	openai "github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/suite"
)

type AudioTranscriptionGeneratorSuite struct {
	suite.Suite
}

func TestAudioTranscriptionGeneratorSuite(t *testing.T) {
	suite.Run(t, new(AudioTranscriptionGeneratorSuite))
}

func (s *AudioTranscriptionGeneratorSuite) TestNewAudioTranscriptionGeneratorEmptyInputReturnsError() {
	generator, err := NewAudioTranscriptionGenerator("   ", model.AudioOptions{})

	s.Require().Error(err)
	s.Nil(generator)
}

func (s *AudioTranscriptionGeneratorSuite) TestResolveAudioTranscriptionModelNameUsesDefault() {
	modelName := resolveAudioTranscriptionModelName(model.AudioOptions{})
	s.Equal(defaultAudioTranscriptionModelName, modelName)
}

func (s *AudioTranscriptionGeneratorSuite) TestResolveAudioTranscriptionModelNameUsesConfigValue() {
	resolved := resolveAudioTranscriptionModelName(model.AudioOptions{
		Model: string(openai.AudioModelGPT4oMiniTranscribe),
	})
	s.Equal(string(openai.AudioModelGPT4oMiniTranscribe), resolved)
}

func (s *AudioTranscriptionGeneratorSuite) TestAudioGeneratorConfigFromOptionsMapsFields() {
	cfg := audioGeneratorConfigFromOptions(model.AudioOptions{
		IgnoreInvalidGeneratorOptions: true,
		URL:                           "https://example.local/v1",
		AuthToken:                     "abc",
		Model:                         "whisper-1",
	})

	s.True(cfg.IgnoreInvalidGeneratorOptions)
	s.Equal("https://example.local/v1", cfg.URL)
	s.Equal("abc", cfg.AuthToken)
	s.Require().NotNil(cfg.Model)
	s.Equal("whisper-1", *cfg.Model)
}

func (s *AudioTranscriptionGeneratorSuite) TestCloneAudioOptionsCopiesKeywords() {
	opts := model.AudioOptions{
		Keywords: map[string]string{
			"afib": "atrial fibrillation",
		},
	}

	cloned := cloneAudioOptions(opts)
	cloned.Keywords["afib"] = "changed"

	s.Equal("atrial fibrillation", opts.Keywords["afib"])
	s.Equal("changed", cloned.Keywords["afib"])
}

func (s *AudioTranscriptionGeneratorSuite) TestBuildWordsToWatchPromptUsesKeywordMapKeys() {
	prompt := buildWordsToWatchPrompt(map[string]string{
		"afib": "atrial fibrillation",
		"htn":  "hypertension",
	})

	s.Equal("afib, htn", prompt)
}

func (s *AudioTranscriptionGeneratorSuite) TestBuildWordsToWatchPromptSkipsEmptyKeys() {
	prompt := buildWordsToWatchPrompt(map[string]string{
		"":           "ignore",
		"  ":         "ignore",
		" creatine ": "creatinine",
	})

	s.Equal("creatine", prompt)
}

func (s *AudioTranscriptionGeneratorSuite) TestApplyOpenAIAudioTranscriptionMetadataUsesTokenUsage() {
	meta := model.GenerationMetadata{}
	response := &openai.AudioTranscriptionNewResponseUnion{
		Usage: openai.AudioTranscriptionNewResponseUnionUsage{
			InputTokens:  10,
			OutputTokens: 5,
			TotalTokens:  15,
			Type:         "tokens",
		},
	}

	applyOpenAIAudioTranscriptionMetadata(meta, response)

	s.Equal("10", meta[model.MetadataKeyInputTokens])
	s.Equal("5", meta[model.MetadataKeyOutputTokens])
	s.Equal("15", meta[model.MetadataKeyTotalTokens])
}

func (s *AudioTranscriptionGeneratorSuite) TestRunAudioTranscriptionInvalidFileReturnsError() {
	c := &client{}

	_, _, err := c.runAudioTranscription(context.Background(), "/path/that/does/not/exist.wav", model.AudioOptions{})

	s.Require().Error(err)
}
