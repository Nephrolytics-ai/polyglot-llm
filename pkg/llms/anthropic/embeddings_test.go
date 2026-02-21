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
	generator, err := NewEmbeddingGenerator("hello")
	s.Nil(generator)
	s.Error(err)
	s.Contains(err.Error(), unsupportedEmbeddingsMessage)
}

func (s *EmbeddingsSuite) TestNewBatchEmbeddingGeneratorReturnsUnsupported() {
	generator, err := NewBatchEmbeddingGenerator([]string{"hello", "world"})
	s.Nil(generator)
	s.Error(err)
	s.Contains(err.Error(), unsupportedEmbeddingsMessage)
}

func (s *EmbeddingsSuite) TestNewBatchEmbeddingGeneratorEmptyInputReturnsError() {
	generator, err := NewBatchEmbeddingGenerator([]string{"hello", "  "})
	s.Nil(generator)
	s.Error(err)
	s.Contains(err.Error(), "input at index 1 is empty")
}
