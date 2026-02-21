package anthropic

import (
	"errors"
	"fmt"
	"strings"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
)

const unsupportedEmbeddingsMessage = "anthropic provider does not currently support embeddings in this library; use another provider"

func NewEmbeddingGenerator(opts ...model.GeneratorOption) (model.EmbeddingGenerator, error) {
	_ = opts
	return nil, utils.WrapIfNotNil(errors.New(unsupportedEmbeddingsMessage))
}

func validateEmbeddingInputs(inputs []string) error {
	if len(inputs) == 0 {
		return utils.WrapIfNotNil(errors.New("at least one input is required"))
	}
	for i, input := range inputs {
		if strings.TrimSpace(input) == "" {
			return utils.WrapIfNotNil(fmt.Errorf("input at index %d is empty", i))
		}
	}
	return nil
}
