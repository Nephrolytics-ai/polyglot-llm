package openai_response

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/logging"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
	"github.com/invopop/jsonschema"
	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

const (
	defaultModelName = "gpt-5-mini"
	maxToolRounds    = 12
)

type toolHandler func(ctx context.Context, args json.RawMessage) (any, error)

type client struct {
	llmCfg    model.LLMConfig
	apiClient openai.Client
}

func NewStructureContentGenerator[T any](prompt string, llmOpts []model.LLMOption, opts ...model.GeneratorOption) (model.ContentGenerator[T], error) {
	const fn = "openai_response.NewStructureContentGenerator"
	if prompt == "" {
		return nil, utils.WrapIfNotNil(errors.New("prompt is required"), fn)
	}

	cfg := model.ResolveGeneratorOpts(opts...)
	c, err := newClient(llmOpts)
	if err != nil {
		return nil, utils.WrapIfNotNil(err, fn)
	}
	return &structuredGenerator[T]{client: c, prompt: prompt, cfg: cfg}, nil
}

func NewStringContentGenerator(prompt string, llmOpts []model.LLMOption, opts ...model.GeneratorOption) (model.ContentGenerator[string], error) {
	const fn = "openai_response.NewStringContentGenerator"
	if prompt == "" {
		return nil, utils.WrapIfNotNil(errors.New("prompt is required"), fn)
	}

	cfg := model.ResolveGeneratorOpts(opts...)
	c, err := newClient(llmOpts)
	if err != nil {
		return nil, utils.WrapIfNotNil(err, fn)
	}
	return &textGenerator{client: c, prompt: prompt, cfg: cfg}, nil
}

func newClient(llmOpts []model.LLMOption) (*client, error) {
	llmCfg := model.ResolveLLMOpts(llmOpts...)
	requestOpts := make([]option.RequestOption, 0, 2)
	if llmCfg.URL != "" {
		requestOpts = append(requestOpts, option.WithBaseURL(llmCfg.URL))
	}
	if llmCfg.AuthToken != "" {
		requestOpts = append(requestOpts, option.WithAPIKey(llmCfg.AuthToken))
	}

	apiClient := openai.NewClient(requestOpts...)
	return &client{llmCfg: llmCfg, apiClient: apiClient}, nil
}

type structuredGenerator[T any] struct {
	client                 *client
	prompt                 string
	cfg                    model.GeneratorConfig
	promptContextMu        sync.RWMutex
	promptContexts         []*model.PromptContext
	promptContextProviders []model.PromptContextProvider
}

func (g *structuredGenerator[T]) AddPromptContext(ctx context.Context, messageType model.ContextMessageType, content string) {
	log := logging.NewLogger(ctx)
	g.promptContextMu.Lock()
	defer g.promptContextMu.Unlock()

	g.promptContexts = append(g.promptContexts, &model.PromptContext{
		MessageType: messageType,
		Content:     content,
	})
	log.Debugf(
		"openai_response.structuredGenerator.AddPromptContext total_contexts=%d",
		len(g.promptContexts),
	)
}

func (g *structuredGenerator[T]) AddPromptContextProvider(ctx context.Context, provider model.PromptContextProvider) {
	if provider == nil {
		return
	}

	g.promptContextMu.Lock()
	defer g.promptContextMu.Unlock()
	g.promptContextProviders = append(g.promptContextProviders, provider)
	logging.NewLogger(ctx).Debugf(
		"openai_response.structuredGenerator.AddPromptContextProvider total_providers=%d",
		len(g.promptContextProviders),
	)
}

func (g *structuredGenerator[T]) Generate(ctx context.Context) (T, error) {
	const fn = "openai_response.structuredGenerator.Generate"
	log := logging.NewLogger(ctx)
	prompt, contextCount, err := g.promptWithContext(ctx)
	if err != nil {
		log.Errorf("%s error: %v", fn, err)
		var zero T
		return zero, utils.WrapIfNotNil(err, fn)
	}
	log.Infof(
		"%s prompt=%q context_count=%d model=%v temperature=%v max_tokens=%v reasoning=%v tools=%d mcp_tools=%d",
		fn,
		prompt,
		contextCount,
		g.cfg.Model,
		g.cfg.Temperature,
		g.cfg.MaxTokens,
		g.cfg.ReasoningLevel,
		len(g.cfg.Tools),
		len(g.cfg.MCPTools),
	)

	schema, err := generateSchema[T]()
	if err != nil {
		log.Errorf("%s error: %v", fn, err)
		var zero T
		return zero, utils.WrapIfNotNil(err)
	}

	textCfg := responses.ResponseTextConfigParam{
		Format: responses.ResponseFormatTextConfigUnionParam{
			OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   "structured_output",
				Schema: schema,
				Strict: openai.Bool(true),
			},
		},
	}

	response, err := g.client.runResponsesFlow(ctx, prompt, g.cfg, &textCfg)
	if err != nil {
		log.Errorf("%s error: %v", fn, err)
		var zero T
		return zero, utils.WrapIfNotNil(err)
	}

	output := strings.TrimSpace(response.OutputText())
	if output == "" {
		err = errors.New("response output is empty")
		log.Errorf("%s error: %v", fn, err)
		var zero T
		return zero, utils.WrapIfNotNil(err)
	}

	var result T
	err = json.Unmarshal([]byte(output), &result)
	if err != nil {
		log.Errorf("%s error: %v", fn, err)
		var zero T
		return zero, utils.WrapIfNotNil(err)
	}

	return result, nil
}

type textGenerator struct {
	client                 *client
	prompt                 string
	cfg                    model.GeneratorConfig
	promptContextMu        sync.RWMutex
	promptContexts         []*model.PromptContext
	promptContextProviders []model.PromptContextProvider
}

func (g *textGenerator) AddPromptContext(ctx context.Context, messageType model.ContextMessageType, content string) {
	log := logging.NewLogger(ctx)
	g.promptContextMu.Lock()
	defer g.promptContextMu.Unlock()

	g.promptContexts = append(g.promptContexts, &model.PromptContext{
		MessageType: messageType,
		Content:     content,
	})
	log.Debugf(
		"openai_response.textGenerator.AddPromptContext total_contexts=%d",
		len(g.promptContexts),
	)
}

func (g *textGenerator) AddPromptContextProvider(ctx context.Context, provider model.PromptContextProvider) {
	if provider == nil {
		return
	}

	g.promptContextMu.Lock()
	defer g.promptContextMu.Unlock()
	g.promptContextProviders = append(g.promptContextProviders, provider)
	logging.NewLogger(ctx).Debugf(
		"openai_response.textGenerator.AddPromptContextProvider total_providers=%d",
		len(g.promptContextProviders),
	)
}

func (g *textGenerator) Generate(ctx context.Context) (string, error) {
	const fn = "openai_response.textGenerator.Generate"
	log := logging.NewLogger(ctx)
	prompt, contextCount, err := g.promptWithContext(ctx)
	if err != nil {
		log.Errorf("%s error: %v", fn, err)
		return "", utils.WrapIfNotNil(err, fn)
	}
	log.Infof(
		"%s prompt=%q context_count=%d model=%v temperature=%v max_tokens=%v reasoning=%v tools=%d mcp_tools=%d",
		fn,
		prompt,
		contextCount,
		g.cfg.Model,
		g.cfg.Temperature,
		g.cfg.MaxTokens,
		g.cfg.ReasoningLevel,
		len(g.cfg.Tools),
		len(g.cfg.MCPTools),
	)

	response, err := g.client.runResponsesFlow(ctx, prompt, g.cfg, nil)
	if err != nil {
		log.Errorf("%s error: %v", fn, err)
		return "", utils.WrapIfNotNil(err)
	}

	return response.OutputText(), nil
}

func (g *structuredGenerator[T]) promptWithContext(ctx context.Context) (string, int, error) {
	const fn = "openai_response.structuredGenerator.promptWithContext"
	g.promptContextMu.RLock()
	contexts := append([]*model.PromptContext(nil), g.promptContexts...)
	providers := append([]model.PromptContextProvider(nil), g.promptContextProviders...)
	g.promptContextMu.RUnlock()

	for _, provider := range providers {
		provided, err := provider.GenerateContext(ctx)
		if err != nil {
			return "", 0, utils.WrapIfNotNil(err, fn)
		}
		contexts = append(contexts, provided...)
	}

	return buildPromptWithContext(g.prompt, contexts)
}

func (g *textGenerator) promptWithContext(ctx context.Context) (string, int, error) {
	const fn = "openai_response.textGenerator.promptWithContext"
	g.promptContextMu.RLock()
	contexts := append([]*model.PromptContext(nil), g.promptContexts...)
	providers := append([]model.PromptContextProvider(nil), g.promptContextProviders...)
	g.promptContextMu.RUnlock()

	for _, provider := range providers {
		provided, err := provider.GenerateContext(ctx)
		if err != nil {
			return "", 0, utils.WrapIfNotNil(err, fn)
		}
		contexts = append(contexts, provided...)
	}

	return buildPromptWithContext(g.prompt, contexts)
}

func buildPromptWithContext(prompt string, contexts []*model.PromptContext) (string, int, error) {
	if len(contexts) == 0 {
		return prompt, 0, nil
	}

	var b strings.Builder
	contextCount := 0
	for _, contextItem := range contexts {
		if contextItem == nil {
			continue
		}

		content := strings.TrimSpace(contextItem.Content)
		if content == "" {
			continue
		}

		messageType := strings.TrimSpace(string(contextItem.MessageType))
		if messageType == "" {
			messageType = string(model.ContextMessageTypeHuman)
		}

		contextCount++
		b.WriteString("[")
		b.WriteString(messageType)
		b.WriteString("]\n")
		b.WriteString(content)
		b.WriteString("\n\n")
	}

	if contextCount == 0 {
		return prompt, 0, nil
	}

	b.WriteString("[prompt]\n")
	b.WriteString(prompt)
	return b.String(), contextCount, nil
}

func (c *client) runResponsesFlow(
	ctx context.Context,
	prompt string,
	cfg model.GeneratorConfig,
	textCfg *responses.ResponseTextConfigParam,
) (*responses.Response, error) {
	const fn = "openai_response.Client.runResponsesFlow"
	log := logging.NewLogger(ctx)

	initialParams, handlers, err := c.buildInitialParams(ctx, prompt, cfg, textCfg)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	response, err := c.apiClient.Responses.New(ctx, initialParams)
	if err != nil {
		log.Errorf("%s error: %v", fn, err)
		return nil, utils.WrapIfNotNil(err)
	}
	if response == nil {
		err = errors.New("responses API returned nil response")
		log.Errorf("%s error: %v", fn, err)
		return nil, utils.WrapIfNotNil(err)
	}

	for round := 0; round < maxToolRounds; round++ {
		calls := extractFunctionCalls(response)
		if len(calls) == 0 {
			return response, nil
		}

		log.Infof("%s tool_round=%d function_calls=%d", fn, round+1, len(calls))
		outputItems := make([]responses.ResponseInputItemUnionParam, 0, len(calls))

		for _, call := range calls {
			handler, ok := handlers[call.Name]
			if !ok {
				err = fmt.Errorf("no tool handler configured for function %q", call.Name)
				log.Errorf("%s error: %v", fn, err)
				return nil, utils.WrapIfNotNil(err)
			}

			result, callErr := handler(ctx, json.RawMessage(call.Arguments))
			if callErr != nil {
				log.Errorf("%s error: %v", fn, callErr)
				return nil, utils.WrapIfNotNil(callErr)
			}

			outputJSON, marshalErr := json.Marshal(result)
			if marshalErr != nil {
				log.Errorf("%s error: %v", fn, marshalErr)
				return nil, utils.WrapIfNotNil(marshalErr)
			}

			outputItems = append(outputItems, responses.ResponseInputItemParamOfFunctionCallOutput(call.CallID, string(outputJSON)))
		}

		nextParams := buildFollowupParams(initialParams, response.ID, outputItems, textCfg)
		response, err = c.apiClient.Responses.New(ctx, nextParams)
		if err != nil {
			log.Errorf("%s error: %v", fn, err)
			return nil, utils.WrapIfNotNil(err)
		}
		if response == nil {
			err = errors.New("responses API returned nil follow-up response")
			log.Errorf("%s error: %v", fn, err)
			return nil, utils.WrapIfNotNil(err)
		}
	}

	err = fmt.Errorf("exceeded tool call loop limit (%d)", maxToolRounds)
	log.Errorf("%s error: %v", fn, err)
	return nil, utils.WrapIfNotNil(err)
}

func (c *client) buildInitialParams(
	ctx context.Context,
	prompt string,
	cfg model.GeneratorConfig,
	textCfg *responses.ResponseTextConfigParam,
) (responses.ResponseNewParams, map[string]toolHandler, error) {
	const fn = "openai_response.Client.buildInitialParams"
	log := logging.NewLogger(ctx)

	modelName := resolveModelName(cfg)
	cfg, err := normalizeGeneratorOptionsForModel(c.llmCfg, modelName, cfg, log)
	if err != nil {
		return responses.ResponseNewParams{}, nil, utils.WrapIfNotNil(err, fn)
	}

	tools, handlers, err := mapLocalTools(cfg.Tools)
	if err != nil {
		return responses.ResponseNewParams{}, nil, utils.WrapIfNotNil(err)
	}

	mcpTools, err := mapMCPTools(cfg.MCPTools)
	if err != nil {
		return responses.ResponseNewParams{}, nil, utils.WrapIfNotNil(err)
	}

	allTools := make([]responses.ToolUnionParam, 0, len(tools)+len(mcpTools))
	allTools = append(allTools, tools...)
	allTools = append(allTools, mcpTools...)

	params := responses.ResponseNewParams{
		Input: responses.ResponseNewParamsInputUnion{
			OfString: openai.String(prompt),
		},
		Model: shared.ResponsesModel(modelName),
		Tools: allTools,
	}

	if cfg.Temperature != nil {
		params.Temperature = openai.Float(*cfg.Temperature)
	}
	if cfg.MaxTokens != nil {
		params.MaxOutputTokens = openai.Int(int64(*cfg.MaxTokens))
	}
	if cfg.ReasoningLevel != nil {
		params.Reasoning = shared.ReasoningParam{
			Effort: mapReasoningLevel(*cfg.ReasoningLevel),
		}
	}
	if textCfg != nil {
		params.Text = *textCfg
	}

	return params, handlers, nil
}

func normalizeGeneratorOptionsForModel(
	llmCfg model.LLMConfig,
	modelName string,
	cfg model.GeneratorConfig,
	log logging.Logger,
) (model.GeneratorConfig, error) {
	reasoningModel := isReasoningModel(modelName)

	if cfg.Temperature != nil && reasoningModel {
		if llmCfg.IgnoreInvalidGeneratorOptions {
			if log != nil {
				log.Warnf("ignoring temperature for reasoning model %q", modelName)
			}
			cfg.Temperature = nil
		} else {
			return cfg, utils.WrapIfNotNil(
				fmt.Errorf("temperature is not supported for reasoning model %q", modelName),
			)
		}
	}

	if cfg.ReasoningLevel != nil && !reasoningModel {
		if llmCfg.IgnoreInvalidGeneratorOptions {
			if log != nil {
				log.Warnf("ignoring reasoning effort for non-reasoning model %q", modelName)
			}
			cfg.ReasoningLevel = nil
		} else {
			return cfg, utils.WrapIfNotNil(
				fmt.Errorf("reasoning effort is not supported for non-reasoning model %q", modelName),
			)
		}
	}

	return cfg, nil
}

func isReasoningModel(modelName string) bool {
	name := strings.ToLower(strings.TrimSpace(modelName))
	if name == "" {
		return false
	}

	return strings.HasPrefix(name, "o1") ||
		strings.HasPrefix(name, "o3") ||
		strings.HasPrefix(name, "o4") ||
		strings.HasPrefix(name, "gpt-5")
}

func buildFollowupParams(
	initial responses.ResponseNewParams,
	previousResponseID string,
	outputItems []responses.ResponseInputItemUnionParam,
	textCfg *responses.ResponseTextConfigParam,
) responses.ResponseNewParams {
	followup := responses.ResponseNewParams{
		Model:              initial.Model,
		Temperature:        initial.Temperature,
		MaxOutputTokens:    initial.MaxOutputTokens,
		Reasoning:          initial.Reasoning,
		Tools:              initial.Tools,
		PreviousResponseID: openai.String(previousResponseID),
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: outputItems,
		},
	}

	if textCfg != nil {
		followup.Text = *textCfg
	}

	return followup
}

func mapLocalTools(tools []model.Tool) ([]responses.ToolUnionParam, map[string]toolHandler, error) {
	const fn = "openai_response.mapLocalTools"

	responseTools := make([]responses.ToolUnionParam, 0, len(tools))
	handlers := make(map[string]toolHandler, len(tools))

	for _, tool := range tools {
		if tool.Name == "" {
			return nil, nil, utils.WrapIfNotNil(errors.New("tool name is required"), fn)
		}
		if tool.Handler == nil {
			return nil, nil, utils.WrapIfNotNil(fmt.Errorf("tool handler is required for %q", tool.Name), fn)
		}
		if _, exists := handlers[tool.Name]; exists {
			return nil, nil, utils.WrapIfNotNil(fmt.Errorf("duplicate tool name %q", tool.Name), fn)
		}

		parameters := map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}
		if tool.InputSchema != nil {
			parameters = map[string]any(tool.InputSchema)
		}

		param := responses.FunctionToolParam{
			Name:       tool.Name,
			Parameters: parameters,
			Strict:     openai.Bool(true),
		}
		if tool.Description != "" {
			param.Description = openai.String(tool.Description)
		}

		responseTools = append(responseTools, responses.ToolUnionParam{
			OfFunction: &param,
		})
		handlers[tool.Name] = tool.Handler
	}

	return responseTools, handlers, nil
}

func mapMCPTools(tools []model.MCPTool) ([]responses.ToolUnionParam, error) {
	const fn = "openai_response.mapMCPTools"

	responseTools := make([]responses.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		if tool.Name == "" {
			return nil, utils.WrapIfNotNil(errors.New("mcp tool name is required"), fn)
		}
		if tool.URL == "" {
			return nil, utils.WrapIfNotNil(fmt.Errorf("mcp tool URL is required for %q", tool.Name), fn)
		}

		authorization, headers := extractAuthorization(tool.HTTPHeaders)

		param := responses.ToolMcpParam{
			ServerLabel: tool.Name,
			ServerURL:   openai.String(tool.URL),
		}
		if authorization != "" {
			param.Authorization = openai.String(authorization)
		}
		if len(headers) > 0 {
			param.Headers = headers
		}
		if len(tool.AllowedTools) > 0 {
			param.AllowedTools = responses.ToolMcpAllowedToolsUnionParam{
				OfMcpAllowedTools: append([]string(nil), tool.AllowedTools...),
			}
		}

		responseTools = append(responseTools, responses.ToolUnionParam{
			OfMcp: &param,
		})
	}

	return responseTools, nil
}

func extractAuthorization(headers map[string]string) (string, map[string]string) {
	filtered := make(map[string]string, len(headers))
	authorization := ""
	for k, v := range headers {
		if strings.EqualFold(k, "Authorization") {
			authorization = v
			continue
		}
		filtered[k] = v
	}

	if len(filtered) == 0 {
		return authorization, nil
	}
	return authorization, filtered
}

func resolveModelName(cfg model.GeneratorConfig) string {
	if cfg.Model != nil {
		modelName := strings.TrimSpace(*cfg.Model)
		if modelName != "" {
			return modelName
		}
	}
	return defaultModelName
}

func mapReasoningLevel(level model.ReasoningLevel) shared.ReasoningEffort {
	switch level {
	case model.ReasoningLevelNone:
		return shared.ReasoningEffortNone
	case model.ReasoningLevelLow:
		return shared.ReasoningEffortLow
	case model.ReasoningLevelMed:
		return shared.ReasoningEffortMedium
	case model.ReasoningLevelHigh:
		return shared.ReasoningEffortHigh
	default:
		return shared.ReasoningEffortMedium
	}
}

func extractFunctionCalls(response *responses.Response) []responses.ResponseFunctionToolCall {
	if response == nil {
		return nil
	}

	calls := make([]responses.ResponseFunctionToolCall, 0)
	for _, item := range response.Output {
		if item.Type != "function_call" {
			continue
		}

		call := item.AsFunctionCall()
		if call.CallID == "" || call.Name == "" {
			continue
		}
		calls = append(calls, call)
	}

	return calls
}

func generateSchema[T any]() (map[string]any, error) {
	const fn = "openai_response.generateSchema"

	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}

	var value T
	schema := reflector.Reflect(value)

	schemaJSON, err := json.Marshal(schema)
	if err != nil {
		return nil, utils.WrapIfNotNil(err, fn)
	}

	var schemaMap map[string]any
	err = json.Unmarshal(schemaJSON, &schemaMap)
	if err != nil {
		return nil, utils.WrapIfNotNil(err, fn)
	}

	return schemaMap, nil
}
