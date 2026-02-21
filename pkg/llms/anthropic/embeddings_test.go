package anthropic

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type EmbeddingsSuite struct {
	suite.Suite
}

func TestEmbeddingsSuite(t *testing.T) {
	suite.Run(t, new(EmbeddingsSuite))
}

func (s *EmbeddingsSuite) TestNewEmbeddingGeneratorReturnsUnsupported() {
	generator, err := NewEmbeddingGenerator()
	s.Nil(generator)
	s.Error(err)
	s.Contains(err.Error(), unsupportedEmbeddingsMessage)
}

func (s *EmbeddingsSuite) TestValidateEmbeddingInputsEmptyInputReturnsError() {
	err := validateEmbeddingInputs([]string{"hello", "  "})
	s.Error(err)
	s.Contains(err.Error(), "input at index 1 is empty")
}

func (s *EmbeddingsSuite) TestValidateEmbeddingInputsMissingInputReturnsError() {
	err := validateEmbeddingInputs(nil)
	s.Error(err)
	s.Contains(err.Error(), "at least one input is required")
}
