package openai_response

import (
	"context"
	"errors"
	"testing"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/suite"
)

type GeneratorOptionValidationSuite struct {
	suite.Suite
}

func TestGeneratorOptionValidationSuite(t *testing.T) {
	suite.Run(t, new(GeneratorOptionValidationSuite))
}

func (s *GeneratorOptionValidationSuite) TestTemperatureOnReasoningModelReturnsErrorWhenStrict() {
	_, err := normalizeGeneratorOptionsForModel(
		"gpt-5-mini",
		model.ResolveGeneratorOpts(
			model.WithIgnoreInvalidGeneratorOptions(false),
			model.WithModel("gpt-5-mini"),
			model.WithTemperature(0.2),
		),
		nil,
	)

	s.Require().Error(err)
	s.Assert().Contains(err.Error(), "temperature is not supported for reasoning model")
}

func (s *GeneratorOptionValidationSuite) TestReasoningOnNonReasoningModelReturnsErrorWhenStrict() {
	_, err := normalizeGeneratorOptionsForModel(
		"gpt-4.1-mini",
		model.ResolveGeneratorOpts(
			model.WithIgnoreInvalidGeneratorOptions(false),
			model.WithModel("gpt-4.1-mini"),
			model.WithReasoningLevel(model.ReasoningLevelLow),
		),
		nil,
	)

	s.Require().Error(err)
	s.Assert().Contains(err.Error(), "reasoning effort is not supported for non-reasoning model")
}

func (s *GeneratorOptionValidationSuite) TestTemperatureOnReasoningModelIsIgnoredWhenConfigured() {
	normalized, err := normalizeGeneratorOptionsForModel(
		"gpt-5-mini",
		model.ResolveGeneratorOpts(
			model.WithIgnoreInvalidGeneratorOptions(true),
			model.WithModel("gpt-5-mini"),
			model.WithTemperature(0.2),
		),
		nil,
	)

	s.Require().NoError(err)
	s.Assert().Nil(normalized.Temperature)
}

func (s *GeneratorOptionValidationSuite) TestReasoningOnNonReasoningModelIsIgnoredWhenConfigured() {
	normalized, err := normalizeGeneratorOptionsForModel(
		"gpt-4.1-mini",
		model.ResolveGeneratorOpts(
			model.WithIgnoreInvalidGeneratorOptions(true),
			model.WithModel("gpt-4.1-mini"),
			model.WithReasoningLevel(model.ReasoningLevelLow),
		),
		nil,
	)

	s.Require().NoError(err)
	s.Assert().Nil(normalized.ReasoningLevel)
}

func (s *GeneratorOptionValidationSuite) TestBuildInputItemsWithContextIncludesPromptContexts() {
	items, contextCount, err := buildInputItemsWithContext("final prompt", []*model.PromptContext{
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
	s.Require().Len(items, 3)
	assertMessageItem(s, items[0], responses.EasyInputMessageRoleSystem, "system content")
	assertMessageItem(s, items[1], responses.EasyInputMessageRoleUser, "rag content")
	assertMessageItem(s, items[2], responses.EasyInputMessageRoleUser, "final prompt")
}

func (s *GeneratorOptionValidationSuite) TestAddPromptContextIsUsedByGeneratorInputBuilder() {
	g := &textGenerator{prompt: "main prompt"}
	g.AddPromptContext(context.Background(), model.ContextMessageTypeSystem, "be concise")

	items, contextCount, err := g.inputItemsWithContext(context.Background())

	s.Require().NoError(err)
	s.Assert().Equal(1, contextCount)
	s.Require().Len(items, 2)
	assertMessageItem(s, items[0], responses.EasyInputMessageRoleSystem, "be concise")
	assertMessageItem(s, items[1], responses.EasyInputMessageRoleUser, "main prompt")
}

func (s *GeneratorOptionValidationSuite) TestAddPromptContextProviderIsCalledDuringInputBuild() {
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

	items, contextCount, err := g.inputItemsWithContext(context.Background())

	s.Require().NoError(err)
	s.Assert().Equal(1, provider.calls)
	s.Assert().Equal(1, contextCount)
	s.Require().Len(items, 2)
	assertMessageItem(s, items[0], responses.EasyInputMessageRoleUser, "provider rag content")
	assertMessageItem(s, items[1], responses.EasyInputMessageRoleUser, "main prompt")
}

func (s *GeneratorOptionValidationSuite) TestInputBuildReturnsProviderError() {
	provider := &stubPromptContextProvider{
		err: errors.New("provider failed"),
	}

	g := &textGenerator{prompt: "main prompt"}
	g.AddPromptContextProvider(context.Background(), provider)

	_, _, err := g.inputItemsWithContext(context.Background())

	s.Require().Error(err)
	s.Assert().Contains(err.Error(), "provider failed")
}

func (s *GeneratorOptionValidationSuite) TestMapContextMessageRole() {
	s.Assert().Equal(responses.EasyInputMessageRoleSystem, mapContextMessageRole(model.ContextMessageTypeSystem))
	s.Assert().Equal(responses.EasyInputMessageRoleAssistant, mapContextMessageRole(model.ContextMessageTypeAssistant))
	s.Assert().Equal(responses.EasyInputMessageRoleUser, mapContextMessageRole(model.ContextMessageTypeHuman))
	s.Assert().Equal(responses.EasyInputMessageRoleUser, mapContextMessageRole(model.ContextMessageType("unknown")))
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

func assertMessageItem(s *GeneratorOptionValidationSuite, item responses.ResponseInputItemUnionParam, expectedRole responses.EasyInputMessageRole, expectedContent string) {
	s.Require().NotNil(item.OfMessage)
	s.Assert().Equal(expectedRole, item.OfMessage.Role)
	s.Assert().Equal(expectedContent, item.OfMessage.Content.OfString.Value)
}
