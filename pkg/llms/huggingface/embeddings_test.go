package huggingface

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

func (s *EmbeddingsSuite) TestValidateEmbeddingInputsEmptySliceReturnsError() {
	err := validateEmbeddingInputs(nil)
	s.Error(err)
	s.Contains(err.Error(), "at least one input is required")
}

func (s *EmbeddingsSuite) TestValidateEmbeddingInputsBlankEntryReturnsError() {
	err := validateEmbeddingInputs([]string{"hello", "  "})
	s.Error(err)
	s.Contains(err.Error(), "input at index 1 is empty")
}

func (s *EmbeddingsSuite) TestValidateEmbeddingInputsSuccess() {
	err := validateEmbeddingInputs([]string{"hello", "world"})
	s.NoError(err)
}

func (s *EmbeddingsSuite) TestNewEmbeddingGeneratorRequiresAuthToken() {
	gen, err := NewEmbeddingGenerator()
	s.Nil(gen)
	s.Error(err)
	s.Contains(err.Error(), "auth token is required")
}
