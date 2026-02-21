package gemini

import (
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

func (s *AudioTranscriptionGeneratorSuite) TestBuildWordsToWatchPromptUsesKeywordMapKeys() {
	prompt := buildWordsToWatchPrompt(map[string]string{
		"egfr": "estimated glomerular filtration rate",
		"afib": "atrial fibrillation",
	})

	s.Equal("afib, egfr", prompt)
}

func (s *AudioTranscriptionGeneratorSuite) TestBuildAudioTranscriptionPromptIncludesKeywords() {
	prompt := buildAudioTranscriptionPrompt(map[string]string{
		"egfr": "estimated glomerular filtration rate",
	})

	s.Contains(prompt, "Transcribe this audio accurately")
	s.Contains(prompt, "egfr")
}
