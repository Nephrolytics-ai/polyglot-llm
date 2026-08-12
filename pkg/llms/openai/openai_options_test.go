package openai

import (
	"context"
	"errors"
	"testing"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
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

func (s *GeneratorOptionValidationSuite) TestWithServiceTierStoresProviderOption() {
	cfg := model.ResolveGeneratorOpts(
		WithServiceTier(ServiceTierFlex),
	)

	s.Require().NotNil(cfg.ProviderOptions)
	s.Equal(ServiceTierFlex, cfg.ProviderOptions[serviceTierProviderOptionKey])
}

func (s *GeneratorOptionValidationSuite) TestResolveServiceTierMapsStandard() {
	tier, err := resolveServiceTier(model.ResolveGeneratorOpts(WithServiceTier(ServiceTierStandard)))

	s.Require().NoError(err)
	s.Require().NotNil(tier)
	s.Equal(responses.ResponseNewParamsServiceTierDefault, *tier)
}

func (s *GeneratorOptionValidationSuite) TestResolveServiceTierMapsAuto() {
	tier, err := resolveServiceTier(model.ResolveGeneratorOpts(WithServiceTier(ServiceTierAuto)))

	s.Require().NoError(err)
	s.Require().NotNil(tier)
	s.Equal(responses.ResponseNewParamsServiceTierAuto, *tier)
}

func (s *GeneratorOptionValidationSuite) TestResolveServiceTierMapsFlex() {
	tier, err := resolveServiceTier(model.ResolveGeneratorOpts(WithServiceTier(ServiceTierFlex)))

	s.Require().NoError(err)
	s.Require().NotNil(tier)
	s.Equal(responses.ResponseNewParamsServiceTierFlex, *tier)
}

func (s *GeneratorOptionValidationSuite) TestResolveServiceTierMapsFast() {
	tier, err := resolveServiceTier(model.ResolveGeneratorOpts(WithServiceTier(ServiceTierFast)))

	s.Require().NoError(err)
	s.Require().NotNil(tier)
	s.Equal(responses.ResponseNewParamsServiceTier("fast"), *tier)
}

func (s *GeneratorOptionValidationSuite) TestResolveServiceTierMapsPriority() {
	tier, err := resolveServiceTier(model.ResolveGeneratorOpts(WithServiceTier(ServiceTierPriority)))

	s.Require().NoError(err)
	s.Require().NotNil(tier)
	s.Equal(responses.ResponseNewParamsServiceTierPriority, *tier)
}

func (s *GeneratorOptionValidationSuite) TestResolveServiceTierReturnsNilWhenUnset() {
	tier, err := resolveServiceTier(model.ResolveGeneratorOpts())

	s.Require().NoError(err)
	s.Nil(tier)
}

func (s *GeneratorOptionValidationSuite) TestResolveServiceTierReturnsErrorForInvalidValue() {
	cfg := model.ResolveGeneratorOpts()
	cfg.ProviderOptions = map[string]any{
		serviceTierProviderOptionKey: ServiceTier("bad-tier"),
	}

	tier, err := resolveServiceTier(cfg)

	s.Require().Error(err)
	s.Nil(tier)
	s.Contains(err.Error(), "unsupported openai service tier")
}

func (s *GeneratorOptionValidationSuite) TestResolvePromptCacheKeyReturnsValueWhenConfigured() {
	key, err := resolvePromptCacheKey(model.ResolveGeneratorOpts(
		model.WithProviderOption(promptCacheKeyProviderOptionKey, "patient-123"),
	))

	s.Require().NoError(err)
	s.Require().NotNil(key)
	s.Equal("patient-123", *key)
}

func (s *GeneratorOptionValidationSuite) TestResolvePromptCacheKeyReturnsNilWhenUnset() {
	key, err := resolvePromptCacheKey(model.ResolveGeneratorOpts())

	s.Require().NoError(err)
	s.Nil(key)
}

func (s *GeneratorOptionValidationSuite) TestResolvePromptCacheKeyReturnsNilForBlankValue() {
	key, err := resolvePromptCacheKey(model.ResolveGeneratorOpts(
		model.WithProviderOption(promptCacheKeyProviderOptionKey, "   "),
	))

	s.Require().NoError(err)
	s.Nil(key)
}

func (s *GeneratorOptionValidationSuite) TestResolvePromptCacheKeyReturnsErrorForInvalidType() {
	key, err := resolvePromptCacheKey(model.ResolveGeneratorOpts(
		model.WithProviderOption(promptCacheKeyProviderOptionKey, 123),
	))

	s.Require().Error(err)
	s.Nil(key)
	s.Contains(err.Error(), "invalid openai prompt cache key option type")
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

func (s *GeneratorOptionValidationSuite) TestMCPHeadersWithAuthTokenAddsAuthorizationWhenMissing() {
	headers := mcpHeadersWithAuthToken(
		map[string]string{"X-Custom": "abc"},
		"mcp-token-123",
	)

	s.Require().NotNil(headers)
	s.Equal("abc", headers["X-Custom"])
	s.Equal("Bearer mcp-token-123", headers["Authorization"])
}

func (s *GeneratorOptionValidationSuite) TestMCPHeadersWithAuthTokenPreservesAuthorizationWhenPresent() {
	headers := mcpHeadersWithAuthToken(
		map[string]string{
			"Authorization": "Bearer existing",
			"X-Custom":      "abc",
		},
		"mcp-token-123",
	)

	s.Require().NotNil(headers)
	s.Equal("Bearer existing", headers["Authorization"])
	s.Equal("abc", headers["X-Custom"])
}

func (s *GeneratorOptionValidationSuite) TestBuildInitialParamsSetsServiceTierWhenConfigured() {
	c := &client{}
	cfg := model.ResolveGeneratorOpts(
		model.WithModel("gpt-5-mini"),
		WithServiceTier(ServiceTierFlex),
	)

	params, handlers, err := c.buildInitialParams(
		context.Background(),
		responses.ResponseNewParamsInputUnion{
			OfInputItemList: responses.ResponseInputParam{
				responses.ResponseInputItemParamOfMessage("hello", responses.EasyInputMessageRoleUser),
			},
		},
		cfg,
		nil,
	)

	s.Require().NoError(err)
	s.Empty(handlers)
	s.Equal(responses.ResponseNewParamsServiceTierFlex, params.ServiceTier)
}

func (s *GeneratorOptionValidationSuite) TestBuildInitialParamsLeavesServiceTierUnsetByDefault() {
	c := &client{}
	cfg := model.ResolveGeneratorOpts(
		model.WithModel("gpt-5-mini"),
	)

	params, _, err := c.buildInitialParams(
		context.Background(),
		responses.ResponseNewParamsInputUnion{
			OfInputItemList: responses.ResponseInputParam{
				responses.ResponseInputItemParamOfMessage("hello", responses.EasyInputMessageRoleUser),
			},
		},
		cfg,
		nil,
	)

	s.Require().NoError(err)
	s.Equal(responses.ResponseNewParamsServiceTier(""), params.ServiceTier)
}

func (s *GeneratorOptionValidationSuite) TestBuildInitialParamsSetsPromptCacheKeyWhenConfigured() {
	c := &client{}
	cfg := model.ResolveGeneratorOpts(
		model.WithModel("gpt-5-mini"),
		model.WithProviderOption(promptCacheKeyProviderOptionKey, "encounter-42"),
	)

	params, handlers, err := c.buildInitialParams(
		context.Background(),
		responses.ResponseNewParamsInputUnion{
			OfInputItemList: responses.ResponseInputParam{
				responses.ResponseInputItemParamOfMessage("hello", responses.EasyInputMessageRoleUser),
			},
		},
		cfg,
		nil,
	)

	s.Require().NoError(err)
	s.Empty(handlers)
	s.True(params.PromptCacheKey.Valid())
	s.Equal("encounter-42", params.PromptCacheKey.Value)
}

func (s *GeneratorOptionValidationSuite) TestBuildInitialParamsLeavesPromptCacheKeyUnsetByDefault() {
	c := &client{}
	cfg := model.ResolveGeneratorOpts(
		model.WithModel("gpt-5-mini"),
	)

	params, _, err := c.buildInitialParams(
		context.Background(),
		responses.ResponseNewParamsInputUnion{
			OfInputItemList: responses.ResponseInputParam{
				responses.ResponseInputItemParamOfMessage("hello", responses.EasyInputMessageRoleUser),
			},
		},
		cfg,
		nil,
	)

	s.Require().NoError(err)
	s.False(params.PromptCacheKey.Valid())
}

func (s *GeneratorOptionValidationSuite) TestBuildInitialParamsAllowsPromptCacheKeyAndServiceTierTogether() {
	c := &client{}
	cfg := model.ResolveGeneratorOpts(
		model.WithModel("gpt-5-mini"),
		model.WithProviderOption(promptCacheKeyProviderOptionKey, "encounter-42"),
		WithServiceTier(ServiceTierAuto),
	)

	params, _, err := c.buildInitialParams(
		context.Background(),
		responses.ResponseNewParamsInputUnion{
			OfInputItemList: responses.ResponseInputParam{
				responses.ResponseInputItemParamOfMessage("hello", responses.EasyInputMessageRoleUser),
			},
		},
		cfg,
		nil,
	)

	s.Require().NoError(err)
	s.Equal(responses.ResponseNewParamsServiceTierAuto, params.ServiceTier)
	s.True(params.PromptCacheKey.Valid())
	s.Equal("encounter-42", params.PromptCacheKey.Value)
}

func (s *GeneratorOptionValidationSuite) TestBuildStatelessFollowupParamsPreservesServiceTier() {
	initial := responses.ResponseNewParams{
		Model:           shared.ResponsesModel("gpt-5-mini"),
		Temperature:     openai.Float(0.2),
		MaxOutputTokens: openai.Int(128),
		ServiceTier:     responses.ResponseNewParamsServiceTierAuto,
	}

	followup := buildStatelessFollowupParams(
		initial,
		responses.ResponseInputParam{
			responses.ResponseInputItemParamOfMessage("hello", responses.EasyInputMessageRoleUser),
		},
		nil,
	)

	s.Equal(responses.ResponseNewParamsServiceTierAuto, followup.ServiceTier)
}

func (s *GeneratorOptionValidationSuite) TestBuildStatelessFollowupParamsPreservesPromptCacheKey() {
	initial := responses.ResponseNewParams{
		Model:          shared.ResponsesModel("gpt-5-mini"),
		PromptCacheKey: openai.String("encounter-42"),
	}

	followup := buildStatelessFollowupParams(
		initial,
		responses.ResponseInputParam{
			responses.ResponseInputItemParamOfMessage("hello", responses.EasyInputMessageRoleUser),
		},
		nil,
	)

	s.True(followup.PromptCacheKey.Valid())
	s.Equal("encounter-42", followup.PromptCacheKey.Value)
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
