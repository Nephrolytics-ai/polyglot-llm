package anthropic

import (
	"testing"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/stretchr/testify/suite"
)

type OptionsSuite struct {
	suite.Suite
}

func TestOptionsSuite(t *testing.T) {
	suite.Run(t, new(OptionsSuite))
}

func (s *OptionsSuite) TestReasoningLevelStrictReturnsError() {
	_, err := normalizeGeneratorOptionsForProvider(
		model.ResolveGeneratorOpts(
			model.WithIgnoreInvalidGeneratorOptions(false),
			model.WithReasoningLevel(model.ReasoningLevelLow),
		),
		nil,
	)

	s.Error(err)
	s.Contains(err.Error(), "reasoning level is not supported")
}

func (s *OptionsSuite) TestReasoningLevelIgnoredWhenConfigured() {
	normalized, err := normalizeGeneratorOptionsForProvider(
		model.ResolveGeneratorOpts(
			model.WithIgnoreInvalidGeneratorOptions(true),
			model.WithReasoningLevel(model.ReasoningLevelLow),
		),
		nil,
	)

	s.NoError(err)
	s.Nil(normalized.ReasoningLevel)
}
