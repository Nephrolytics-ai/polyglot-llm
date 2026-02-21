package model

import "context"

type AudioOptions struct {
	IgnoreInvalidGeneratorOptions bool
	URL                           string
	AuthToken                     string
	Model                         string
	// keywords to watch for in the transcript.  The key, is the word you want, the sting is a commoa sperated list of common mistypes of the word to watch for.
	//Not all models will handle this the same
	Keywords map[string]string
}

// NewAudioTranscriptionGeneratorFunc creates an audio transcription generator for a source file.
type NewAudioTranscriptionGeneratorFunc func(filePath string, opts AudioOptions) (AudioTranscriptionGenerator, error)

// AudioTranscriptionGenerator represents "audio file in, transcript out".
type AudioTranscriptionGenerator interface {
	Generate(ctx context.Context) (string, GenerationMetadata, error)
}
