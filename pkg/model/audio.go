package model

type AudioOptions struct {
	IgnoreInvalidGeneratorOptions bool
	URL                           string
	AuthToken                     string
	Model                         string
	// keywords to watch for in the transcript.  The key, is the word you want, the string is a comma separated list of common mistypes of the word to watch for.
	//Not all models will handle this the same
	Keywords map[string]string
}
