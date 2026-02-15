package openai_response

import (
	"context"
	"errors"
	"testing"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/stretchr/testify/suite"
)

type GeneratorOptionValidationSuite struct {
	suite.Suite
}

func TestGeneratorOptionValidationSuite(t *testing.T) {
	suite.Run(t, new(GeneratorOptionValidationSuite))
}

func (s *GeneratorOptionValidationSuite) TestTemperatureOnReasoningModelReturnsErrorWhenStrict() {
	cfg := model.ResolveGeneratorOpts(
		model.WithModel("gpt-5-mini"),
		model.WithTemperature(0.2),
	)

	_, err := normalizeGeneratorOptionsForModel(
		model.ResolveLLMOpts(model.WithIgnoreInvalidGeneratorOptions(false)),
		"gpt-5-mini",
		cfg,
		nil,
	)

	s.Require().Error(err)
	s.Assert().Contains(err.Error(), "temperature is not supported for reasoning model")
}

func (s *GeneratorOptionValidationSuite) TestReasoningOnNonReasoningModelReturnsErrorWhenStrict() {
	cfg := model.ResolveGeneratorOpts(
		model.WithModel("gpt-4.1-mini"),
		model.WithReasoningLevel(model.ReasoningLevelLow),
	)

	_, err := normalizeGeneratorOptionsForModel(
		model.ResolveLLMOpts(model.WithIgnoreInvalidGeneratorOptions(false)),
		"gpt-4.1-mini",
		cfg,
		nil,
	)

	s.Require().Error(err)
	s.Assert().Contains(err.Error(), "reasoning effort is not supported for non-reasoning model")
}

func (s *GeneratorOptionValidationSuite) TestTemperatureOnReasoningModelIsIgnoredWhenConfigured() {
	cfg := model.ResolveGeneratorOpts(
		model.WithModel("gpt-5-mini"),
		model.WithTemperature(0.2),
	)

	normalized, err := normalizeGeneratorOptionsForModel(
		model.ResolveLLMOpts(model.WithIgnoreInvalidGeneratorOptions(true)),
		"gpt-5-mini",
		cfg,
		nil,
	)

	s.Require().NoError(err)
	s.Assert().Nil(normalized.Temperature)
}

func (s *GeneratorOptionValidationSuite) TestReasoningOnNonReasoningModelIsIgnoredWhenConfigured() {
	cfg := model.ResolveGeneratorOpts(
		model.WithModel("gpt-4.1-mini"),
		model.WithReasoningLevel(model.ReasoningLevelLow),
	)

	normalized, err := normalizeGeneratorOptionsForModel(
		model.ResolveLLMOpts(model.WithIgnoreInvalidGeneratorOptions(true)),
		"gpt-4.1-mini",
		cfg,
		nil,
	)

	s.Require().NoError(err)
	s.Assert().Nil(normalized.ReasoningLevel)
}

func (s *GeneratorOptionValidationSuite) TestBuildPromptWithContextIncludesPromptContexts() {
	result, contextCount, err := buildPromptWithContext("final prompt", []*model.PromptContext{
		{
			MessageType: model.ContextMessageTypeSystem,
			Content:     "system content",
		},
		{
			MessageType: model.ContextMessageTypeHuman,
			Content:     "rag content",
		},
	})

	s.Require().NoError(err)
	s.Assert().Equal(2, contextCount)
	s.Assert().Contains(result, "[system]\nsystem content")
	s.Assert().Contains(result, "[human]\nrag content")
	s.Assert().Contains(result, "[prompt]\nfinal prompt")
}

func (s *GeneratorOptionValidationSuite) TestAddPromptContextIsUsedByGeneratorPromptBuilder() {
	g := &textGenerator{prompt: "main prompt"}
	g.AddPromptContext(context.Background(), model.ContextMessageTypeSystem, "be concise")

	prompt, contextCount, err := g.promptWithContext(context.Background())

	s.Require().NoError(err)
	s.Assert().Equal(1, contextCount)
	s.Assert().Contains(prompt, "[system]\nbe concise")
	s.Assert().Contains(prompt, "[prompt]\nmain prompt")
}

func (s *GeneratorOptionValidationSuite) TestAddPromptContextProviderIsCalledDuringPromptBuild() {
	provider := &stubPromptContextProvider{
		contexts: []*model.PromptContext{
			{
				MessageType: model.ContextMessageTypeHuman,
				Content:     "provider rag content",
			},
		},
	}

	g := &textGenerator{prompt: "main prompt"}
	g.AddPromptContextProvider(context.Background(), provider)

	prompt, contextCount, err := g.promptWithContext(context.Background())

	s.Require().NoError(err)
	s.Assert().Equal(1, provider.calls)
	s.Assert().Equal(1, contextCount)
	s.Assert().Contains(prompt, "[human]\nprovider rag content")
	s.Assert().Contains(prompt, "[prompt]\nmain prompt")
}

func (s *GeneratorOptionValidationSuite) TestPromptBuildReturnsProviderError() {
	provider := &stubPromptContextProvider{
		err: errors.New("provider failed"),
	}

	g := &textGenerator{prompt: "main prompt"}
	g.AddPromptContextProvider(context.Background(), provider)

	_, _, err := g.promptWithContext(context.Background())

	s.Require().Error(err)
	s.Assert().Contains(err.Error(), "provider failed")
}

type stubPromptContextProvider struct {
	calls    int
	contexts []*model.PromptContext
	err      error
}

func (s *stubPromptContextProvider) GenerateContext(ctx context.Context) ([]*model.PromptContext, error) {
	s.calls++
	if s.err != nil {
		return nil, s.err
	}
	return s.contexts, nil
}
