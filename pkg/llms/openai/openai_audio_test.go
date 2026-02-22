package openai

import (
	"context"
	"encoding/json"
	"strings"
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
		Keywords: []model.AudioKeyword{
			{
				Word:           "afib",
				CommonMistypes: []string{"a fib", "afibb"},
				Definition:     "Atrial fibrillation.",
			},
		},
	}

	cloned := cloneAudioOptions(opts)
	cloned.Keywords[0].Word = "changed"
	cloned.Keywords[0].CommonMistypes[0] = "changed-mistype"
	cloned.Keywords[0].Definition = "changed-definition"

	s.Equal("afib", opts.Keywords[0].Word)
	s.Equal("a fib", opts.Keywords[0].CommonMistypes[0])
	s.Equal("Atrial fibrillation.", opts.Keywords[0].Definition)
}

func (s *AudioTranscriptionGeneratorSuite) TestBuildCommonMissedWordsPromptUsesKeywordStructs() {
	prompt, err := buildCommonMissedWordsPrompt([]model.AudioKeyword{
		{
			Word:           "losartan",
			CommonMistypes: []string{"losartan potassium", "losarton"},
			Definition:     "An angiotensin II receptor blocker (ARB) used to treat high blood pressure.",
		},
	})
	s.Require().NoError(err)

	s.Equal(
		`Common missed words: [{"word":"losartan","common_mistypes":["losartan potassium","losarton"],"definition":"An angiotensin II receptor blocker (ARB) used to treat high blood pressure."}]`,
		prompt,
	)
}

func (s *AudioTranscriptionGeneratorSuite) TestBuildCommonMissedWordsPromptSkipsEmptyKeywordEntries() {
	prompt, err := buildCommonMissedWordsPrompt([]model.AudioKeyword{
		{},
		{
			Word:           " creatinine ",
			CommonMistypes: []string{" ", "creatnine"},
		},
	})
	s.Require().NoError(err)

	payload := strings.TrimPrefix(prompt, "Common missed words: ")
	var parsed []model.AudioKeyword
	s.Require().NoError(json.Unmarshal([]byte(payload), &parsed))
	s.Require().Len(parsed, 1)
	s.Equal("creatinine", parsed[0].Word)
	s.Equal([]string{"creatnine"}, parsed[0].CommonMistypes)
}

func (s *AudioTranscriptionGeneratorSuite) TestBuildAudioTranscriptionPromptUsesCustomPrompt() {
	prompt, err := buildAudioTranscriptionPrompt(model.AudioOptions{
		Prompt: "Use this exact audio prompt.",
		Keywords: []model.AudioKeyword{
			{Word: "should-not-appear"},
		},
	})
	s.Require().NoError(err)
	s.Equal("Use this exact audio prompt.", prompt)
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
