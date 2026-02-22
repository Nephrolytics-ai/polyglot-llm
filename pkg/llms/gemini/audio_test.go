package gemini

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
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
	s.Equal(defaultGenerationModelName, modelName)
}

func (s *AudioTranscriptionGeneratorSuite) TestResolveAudioTranscriptionModelNameUsesConfigValue() {
	resolved := resolveAudioTranscriptionModelName(model.AudioOptions{
		Model: "gemini-2.5-flash",
	})
	s.Equal("gemini-2.5-flash", resolved)
}

func (s *AudioTranscriptionGeneratorSuite) TestResolveAudioMIMETypeUsesCommonMappings() {
	mimeType, err := resolveAudioMIMEType("example.m4a")
	s.Require().NoError(err)
	s.Equal("audio/mp4", mimeType)
}

func (s *AudioTranscriptionGeneratorSuite) TestResolveAudioMIMETypeUnsupportedExtensionReturnsError() {
	_, err := resolveAudioMIMEType("example.txt")
	s.Require().Error(err)
	s.Contains(err.Error(), "unsupported audio")
}

func (s *AudioTranscriptionGeneratorSuite) TestBuildCommonMissedWordsPromptUsesKeywordStructs() {
	prompt, err := buildCommonMissedWordsPrompt([]model.AudioKeyword{
		{
			Word:           "losartan",
			CommonMistypes: []string{"losarton"},
			Definition:     "An angiotensin II receptor blocker (ARB) used to treat high blood pressure.",
		},
	})
	s.Require().NoError(err)

	s.Equal(
		`Common missed words: [{"Word":"losartan","CommonMistypes":["losarton"],"Definition":"An angiotensin II receptor blocker (ARB) used to treat high blood pressure."}]`,
		prompt,
	)
}

func (s *AudioTranscriptionGeneratorSuite) TestBuildAudioTranscriptionPromptIncludesKeywords() {
	prompt, err := buildAudioTranscriptionPrompt(model.AudioOptions{
		Keywords: []model.AudioKeyword{
			{
				Word:           "egfr",
				CommonMistypes: []string{"e g f r"},
				Definition:     "Estimated glomerular filtration rate.",
			},
		},
	})
	s.Require().NoError(err)

	s.Contains(prompt, "Transcribe this audio accurately")
	s.Contains(prompt, "Common missed words:")
	s.Contains(prompt, "\"Word\":\"egfr\"")
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
