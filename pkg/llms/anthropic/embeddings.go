package anthropic

import (
	"errors"
	"fmt"
	"strings"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
)

const unsupportedEmbeddingsMessage = "anthropic provider does not currently support embeddings in this library; use another provider"

func NewEmbeddingGenerator(input string, opts ...model.GeneratorOption) (model.EmbeddingGenerator, error) {
	if strings.TrimSpace(input) == "" {
		return nil, utils.WrapIfNotNil(errors.New("input is required"))
	}
	return nil, utils.WrapIfNotNil(errors.New(unsupportedEmbeddingsMessage))
}

func NewBatchEmbeddingGenerator(inputs []string, opts ...model.GeneratorOption) (model.EmbeddingGenerator, error) {
	if len(inputs) == 0 {
		return nil, utils.WrapIfNotNil(errors.New("at least one input is required"))
	}
	for i, input := range inputs {
		if strings.TrimSpace(input) == "" {
			return nil, utils.WrapIfNotNil(fmt.Errorf("input at index %d is empty", i))
		}
	}
	return nil, utils.WrapIfNotNil(errors.New(unsupportedEmbeddingsMessage))
}
