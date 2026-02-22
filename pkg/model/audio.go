package model

import "context"

type AudioKeyword struct {
	Word           string   `json:"word,omitempty"`
	CommonMistypes []string `json:"common_mistypes,omitempty"`
	Definition     string   `json:"definition,omitempty"`
}


type AudioOptions struct {
	IgnoreInvalidGeneratorOptions bool
	URL                           string
	AuthToken                     string
	Model                         string
	// Prompt optionally overrides the provider's default audio prompt behavior.
	// When Prompt is set, keyword hints are not appended.
	Prompt string
	// Keywords provides domain terms that may be missed in transcription.
	// Providers may convert this into: "Common missed words: <json>"
	// when Prompt is empty.
	Keywords []AudioKeyword
}
