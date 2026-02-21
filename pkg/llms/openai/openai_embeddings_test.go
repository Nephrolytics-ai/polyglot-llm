package openai

import (
	"testing"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	openai "github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/suite"
)

type EmbeddingGeneratorSuite struct {
	suite.Suite
}

func TestEmbeddingGeneratorSuite(t *testing.T) {
	suite.Run(t, new(EmbeddingGeneratorSuite))
}

func (s *EmbeddingGeneratorSuite) TestNewEmbeddingGeneratorNoInputReturnsGenerator() {
	generator, err := NewEmbeddingGenerator()
	s.Require().NoError(err)
	s.NotNil(generator)
}

func (s *EmbeddingGeneratorSuite) TestResolveEmbeddingModelNameUsesDefault() {
	modelName := resolveEmbeddingModelName(model.GeneratorConfig{})
	s.Equal(defaultEmbeddingModelName, modelName)
}

func (s *EmbeddingGeneratorSuite) TestConvertEmbeddingResponseOrdersByIndex() {
	response := &openai.CreateEmbeddingResponse{
		Data: []openai.Embedding{
			{
				Index:     1,
				Embedding: []float64{1.5, 2.5},
			},
			{
				Index:     0,
				Embedding: []float64{3.5, 4.5},
			},
		},
	}

	vectors, err := convertEmbeddingResponse(response, 2)
	s.Require().NoError(err)
	s.Require().Len(vectors, 2)
	s.Require().Len(vectors[0], 2)
	s.Require().Len(vectors[1], 2)

	s.Equal(float64(3.5), vectors[0][0])
	s.Equal(float64(4.5), vectors[0][1])
	s.Equal(float64(1.5), vectors[1][0])
	s.Equal(float64(2.5), vectors[1][1])
}

func (s *EmbeddingGeneratorSuite) TestConvertEmbeddingResponseMismatchedLengthReturnsError() {
	response := &openai.CreateEmbeddingResponse{
		Data: []openai.Embedding{
			{
				Index:     0,
				Embedding: []float64{1.5},
			},
		},
	}

	_, err := convertEmbeddingResponse(response, 2)
	s.Require().Error(err)
	s.Contains(err.Error(), "embedding response size mismatch")
}

func (s *EmbeddingGeneratorSuite) TestValidateEmbeddingInputsEmptyInputsReturnsError() {
	err := validateEmbeddingInputs(nil)
	s.Require().Error(err)
	s.Contains(err.Error(), "at least one input is required")
}

func (s *EmbeddingGeneratorSuite) TestValidateEmbeddingInputsContainsEmptyInputReturnsError() {
	err := validateEmbeddingInputs([]string{"hello", " "})
	s.Require().Error(err)
	s.Contains(err.Error(), "input at index 1 is empty")
}
