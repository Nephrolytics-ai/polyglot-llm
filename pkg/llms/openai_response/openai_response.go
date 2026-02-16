package openai_response

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/logging"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/mcp"
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
	providerName     = "openai_response"
)

type toolHandler func(ctx context.Context, args json.RawMessage) (any, error)

type flowUsageTotals struct {
	APICalls          int
	ToolRounds        int
	InputTokens       int64
	OutputTokens      int64
	TotalTokens       int64
	CachedInputTokens int64
	ReasoningTokens   int64
}

type client struct {
	apiClient openai.Client
}

func NewStructureContentGenerator[T any](prompt string, opts ...model.GeneratorOption) (model.ContentGenerator[T], error) {
	if prompt == "" {
		return nil, utils.WrapIfNotNil(errors.New("prompt is required"))
	}

	cfg := model.ResolveGeneratorOpts(opts...)
	c, err := newClient(cfg)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}
	return &structuredGenerator[T]{client: c, prompt: prompt, cfg: cfg}, nil
}

func NewStringContentGenerator(prompt string, opts ...model.GeneratorOption) (model.ContentGenerator[string], error) {
	if prompt == "" {
		return nil, utils.WrapIfNotNil(errors.New("prompt is required"))
	}

	cfg := model.ResolveGeneratorOpts(opts...)
	c, err := newClient(cfg)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}
	return &textGenerator{client: c, prompt: prompt, cfg: cfg}, nil
}

func newClient(cfg model.GeneratorConfig) (*client, error) {
	requestOpts := make([]option.RequestOption, 0, 2)
	if cfg.URL != "" {
		requestOpts = append(requestOpts, option.WithBaseURL(cfg.URL))
	}
	if cfg.AuthToken != "" {
		requestOpts = append(requestOpts, option.WithAPIKey(cfg.AuthToken))
	}

	apiClient := openai.NewClient(requestOpts...)
	return &client{apiClient: apiClient}, nil
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

func (g *structuredGenerator[T]) Generate(ctx context.Context) (T, model.GenerationMetadata, error) {
	start := time.Now()
	meta := initMetadata(providerName, resolveModelName(g.cfg))
	defer setLatencyMetadata(meta, start)

	log := logging.NewLogger(ctx)
	inputItems, contextCount, err := g.inputItemsWithContext(ctx)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}
	log.Infof(
		"prompt=%q context_count=%d input_items=%d model=%v temperature=%v max_tokens=%v reasoning=%v tools=%d mcp_tools=%d",
		g.prompt,
		contextCount,
		len(inputItems),
		g.cfg.Model,
		g.cfg.Temperature,
		g.cfg.MaxTokens,
		g.cfg.ReasoningLevel,
		len(g.cfg.Tools),
		len(g.cfg.MCPTools),
	)

	schema, err := generateSchema[T]()
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
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

	response, totals, err := g.client.runResponsesFlow(
		ctx,
		responses.ResponseNewParamsInputUnion{
			OfInputItemList: inputItems,
		},
		g.cfg,
		&textCfg,
	)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}
	applyOpenAIResponseMetadata(meta, response, totals)

	output := strings.TrimSpace(response.OutputText())
	if output == "" {
		err = errors.New("response output is empty")
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	var result T
	err = json.Unmarshal([]byte(output), &result)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	return result, meta, nil
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

func (g *textGenerator) Generate(ctx context.Context) (string, model.GenerationMetadata, error) {
	start := time.Now()
	meta := initMetadata(providerName, resolveModelName(g.cfg))
	defer setLatencyMetadata(meta, start)

	log := logging.NewLogger(ctx)
	inputItems, contextCount, err := g.inputItemsWithContext(ctx)
	if err != nil {
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}
	log.Infof(
		"prompt=%q context_count=%d input_items=%d model=%v temperature=%v max_tokens=%v reasoning=%v tools=%d mcp_tools=%d",
		g.prompt,
		contextCount,
		len(inputItems),
		g.cfg.Model,
		g.cfg.Temperature,
		g.cfg.MaxTokens,
		g.cfg.ReasoningLevel,
		len(g.cfg.Tools),
		len(g.cfg.MCPTools),
	)

	response, totals, err := g.client.runResponsesFlow(
		ctx,
		responses.ResponseNewParamsInputUnion{
			OfInputItemList: inputItems,
		},
		g.cfg,
		nil,
	)
	if err != nil {
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}
	applyOpenAIResponseMetadata(meta, response, totals)

	return response.OutputText(), meta, nil
}

func (g *structuredGenerator[T]) inputItemsWithContext(ctx context.Context) (responses.ResponseInputParam, int, error) {
	g.promptContextMu.RLock()
	contexts := append([]*model.PromptContext(nil), g.promptContexts...)
	providers := append([]model.PromptContextProvider(nil), g.promptContextProviders...)
	g.promptContextMu.RUnlock()

	for _, provider := range providers {
		provided, err := provider.GenerateContext(ctx)
		if err != nil {
			return nil, 0, utils.WrapIfNotNil(err)
		}
		contexts = append(contexts, provided...)
	}

	return buildInputItemsWithContext(g.prompt, contexts)
}

func (g *textGenerator) inputItemsWithContext(ctx context.Context) (responses.ResponseInputParam, int, error) {
	g.promptContextMu.RLock()
	contexts := append([]*model.PromptContext(nil), g.promptContexts...)
	providers := append([]model.PromptContextProvider(nil), g.promptContextProviders...)
	g.promptContextMu.RUnlock()

	for _, provider := range providers {
		provided, err := provider.GenerateContext(ctx)
		if err != nil {
			return nil, 0, utils.WrapIfNotNil(err)
		}
		contexts = append(contexts, provided...)
	}

	return buildInputItemsWithContext(g.prompt, contexts)
}

func buildInputItemsWithContext(prompt string, contexts []*model.PromptContext) (responses.ResponseInputParam, int, error) {
	items := make(responses.ResponseInputParam, 0, len(contexts)+1)
	contextCount := 0
	for _, contextItem := range contexts {
		if contextItem == nil {
			continue
		}

		content := strings.TrimSpace(contextItem.Content)
		if content == "" {
			continue
		}

		contextCount++
		items = append(
			items,
			responses.ResponseInputItemParamOfMessage(
				content,
				mapContextMessageRole(contextItem.MessageType),
			),
		)
	}

	items = append(
		items,
		responses.ResponseInputItemParamOfMessage(
			prompt,
			responses.EasyInputMessageRoleUser,
		),
	)
	return items, contextCount, nil
}

func (c *client) runResponsesFlow(
	ctx context.Context,
	input responses.ResponseNewParamsInputUnion,
	cfg model.GeneratorConfig,
	textCfg *responses.ResponseTextConfigParam,
) (*responses.Response, flowUsageTotals, error) {
	log := logging.NewLogger(ctx)
	totals := flowUsageTotals{}

	initialParams, handlers, err := c.buildInitialParams(ctx, input, cfg, textCfg)
	if err != nil {
		return nil, totals, utils.WrapIfNotNil(err)
	}
	history, err := seedInputHistory(initialParams.Input)
	if err != nil {
		return nil, totals, utils.WrapIfNotNil(err)
	}

	response, err := c.apiClient.Responses.New(ctx, initialParams)
	if err != nil {
		log.Errorf("error: %v", err)
		return nil, totals, utils.WrapIfNotNil(err)
	}
	if response == nil {
		err = errors.New("responses API returned nil response")
		log.Errorf("error: %v", err)
		return nil, totals, utils.WrapIfNotNil(err)
	}
	accumulateFlowUsage(&totals, response)

	for round := 0; round < maxToolRounds; round++ {
		priorItems, err := responseOutputToInputItems(response.Output)
		if err != nil {
			log.Errorf("error: %v", err)
			return nil, totals, utils.WrapIfNotNil(err)
		}
		history = append(history, priorItems...)

		calls := extractFunctionCalls(response)
		if len(calls) == 0 {
			return response, totals, nil
		}
		totals.ToolRounds = round + 1

		log.Infof("tool_round=%d function_calls=%d history_items=%d", round+1, len(calls), len(history))
		outputItems := make([]responses.ResponseInputItemUnionParam, 0, len(calls))

		for _, call := range calls {
			handler, ok := handlers[call.Name]
			if !ok {
				err = fmt.Errorf("no tool handler configured for function %q", call.Name)
				log.Errorf("error: %v", err)
				return nil, totals, utils.WrapIfNotNil(err)
			}

			result, callErr := handler(ctx, json.RawMessage(call.Arguments))
			if callErr != nil {
				log.Errorf("error: %v", callErr)
				return nil, totals, utils.WrapIfNotNil(callErr)
			}

			outputJSON, marshalErr := json.Marshal(result)
			if marshalErr != nil {
				log.Errorf("error: %v", marshalErr)
				return nil, totals, utils.WrapIfNotNil(marshalErr)
			}

			outputItems = append(outputItems, responses.ResponseInputItemParamOfFunctionCallOutput(call.CallID, string(outputJSON)))
		}

		history = append(history, outputItems...)
		nextParams := buildStatelessFollowupParams(initialParams, history, textCfg)
		response, err = c.apiClient.Responses.New(ctx, nextParams)
		if err != nil {
			log.Errorf("error: %v", err)
			return nil, totals, utils.WrapIfNotNil(err)
		}
		if response == nil {
			err = errors.New("responses API returned nil follow-up response")
			log.Errorf("error: %v", err)
			return nil, totals, utils.WrapIfNotNil(err)
		}
		accumulateFlowUsage(&totals, response)
	}

	err = fmt.Errorf("exceeded tool call loop limit (%d)", maxToolRounds)
	log.Errorf("error: %v", err)
	return nil, totals, utils.WrapIfNotNil(err)
}

func (c *client) buildInitialParams(
	ctx context.Context,
	input responses.ResponseNewParamsInputUnion,
	cfg model.GeneratorConfig,
	textCfg *responses.ResponseTextConfigParam,
) (responses.ResponseNewParams, map[string]toolHandler, error) {
	log := logging.NewLogger(ctx)

	modelName := resolveModelName(cfg)
	reasoningModel := isReasoningModel(modelName)
	cfg, err := normalizeGeneratorOptionsForModel(modelName, cfg, log)
	if err != nil {
		return responses.ResponseNewParams{}, nil, utils.WrapIfNotNil(err)
	}

	tools, handlers, err := mapLocalTools(cfg.Tools)
	if err != nil {
		return responses.ResponseNewParams{}, nil, utils.WrapIfNotNil(err)
	}

	mcpTools, err := mapMCPTools(ctx, cfg.MCPTools)
	if err != nil {
		return responses.ResponseNewParams{}, nil, utils.WrapIfNotNil(err)
	}

	allTools := make([]responses.ToolUnionParam, 0, len(tools)+len(mcpTools))
	allTools = append(allTools, tools...)
	allTools = append(allTools, mcpTools...)

	params := responses.ResponseNewParams{
		Input: input,
		Model: shared.ResponsesModel(modelName),
		Tools: allTools,
	}
	if reasoningModel {
		params.Include = append(params.Include, responses.ResponseIncludableReasoningEncryptedContent)
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

func mapContextMessageRole(messageType model.ContextMessageType) responses.EasyInputMessageRole {
	switch messageType {
	case model.ContextMessageTypeSystem:
		return responses.EasyInputMessageRoleSystem
	case model.ContextMessageTypeAssistant:
		return responses.EasyInputMessageRoleAssistant
	case model.ContextMessageTypeHuman:
		return responses.EasyInputMessageRoleUser
	default:
		return responses.EasyInputMessageRoleUser
	}
}

func initMetadata(provider string, modelName string) model.GenerationMetadata {
	if strings.TrimSpace(modelName) == "" {
		modelName = "unknown"
	}

	meta := model.GenerationMetadata{
		model.MetadataKeyProvider: provider,
		model.MetadataKeyModel:    modelName,
	}
	return meta
}

func setLatencyMetadata(meta model.GenerationMetadata, start time.Time) {
	if meta == nil {
		return
	}
	meta[model.MetadataKeyLatencyMs] = strconv.FormatInt(time.Since(start).Milliseconds(), 10)
}

func applyOpenAIResponseMetadata(meta model.GenerationMetadata, response *responses.Response, totals flowUsageTotals) {
	if meta == nil {
		return
	}

	meta[model.MetadataKeyAPICalls] = strconv.Itoa(totals.APICalls)
	meta[model.MetadataKeyToolRounds] = strconv.Itoa(totals.ToolRounds)
	meta[model.MetadataKeyInputTokens] = strconv.FormatInt(totals.InputTokens, 10)
	meta[model.MetadataKeyOutputTokens] = strconv.FormatInt(totals.OutputTokens, 10)
	meta[model.MetadataKeyTotalTokens] = strconv.FormatInt(totals.TotalTokens, 10)
	meta[model.MetadataKeyCachedInputTokens] = strconv.FormatInt(totals.CachedInputTokens, 10)
	meta[model.MetadataKeyReasoningTokens] = strconv.FormatInt(totals.ReasoningTokens, 10)
	if response != nil {
		if response.ID != "" {
			meta[model.MetadataKeyResponseID] = response.ID
		}
		if response.Status != "" {
			meta[model.MetadataKeyResponseStatus] = string(response.Status)
		}
	}
}

func accumulateFlowUsage(totals *flowUsageTotals, response *responses.Response) {
	if totals == nil || response == nil {
		return
	}

	totals.APICalls++
	totals.InputTokens += response.Usage.InputTokens
	totals.OutputTokens += response.Usage.OutputTokens
	totals.TotalTokens += response.Usage.TotalTokens
	totals.CachedInputTokens += response.Usage.InputTokensDetails.CachedTokens
	totals.ReasoningTokens += response.Usage.OutputTokensDetails.ReasoningTokens
}

func normalizeGeneratorOptionsForModel(
	modelName string,
	cfg model.GeneratorConfig,
	log logging.Logger,
) (model.GeneratorConfig, error) {
	reasoningModel := isReasoningModel(modelName)

	if cfg.Temperature != nil && reasoningModel {
		if cfg.IgnoreInvalidGeneratorOptions {
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
		if cfg.IgnoreInvalidGeneratorOptions {
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

func buildStatelessFollowupParams(
	initial responses.ResponseNewParams,
	history responses.ResponseInputParam,
	textCfg *responses.ResponseTextConfigParam,
) responses.ResponseNewParams {
	followup := responses.ResponseNewParams{
		Model:           initial.Model,
		Temperature:     initial.Temperature,
		MaxOutputTokens: initial.MaxOutputTokens,
		Reasoning:       initial.Reasoning,
		Tools:           initial.Tools,
		Include:         append([]responses.ResponseIncludable(nil), initial.Include...),
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: append(responses.ResponseInputParam(nil), history...),
		},
	}

	if textCfg != nil {
		followup.Text = *textCfg
	}

	return followup
}

func seedInputHistory(input responses.ResponseNewParamsInputUnion) (responses.ResponseInputParam, error) {
	if len(input.OfInputItemList) > 0 {
		return append(responses.ResponseInputParam(nil), input.OfInputItemList...), nil
	}
	if input.OfString.Valid() {
		return responses.ResponseInputParam{
			responses.ResponseInputItemParamOfMessage(input.OfString.Value, responses.EasyInputMessageRoleUser),
		}, nil
	}
	return nil, utils.WrapIfNotNil(errors.New("response input is empty"))
}

func responseOutputToInputItems(output []responses.ResponseOutputItemUnion) (responses.ResponseInputParam, error) {
	items := make(responses.ResponseInputParam, 0, len(output))
	for _, outputItem := range output {
		var inputItem responses.ResponseInputItemUnion
		err := json.Unmarshal([]byte(outputItem.RawJSON()), &inputItem)
		if err != nil {
			return nil, utils.WrapIfNotNil(err)
		}
		items = append(items, inputItem.ToParam())
	}
	return items, nil
}

func mapLocalTools(tools []model.Tool) ([]responses.ToolUnionParam, map[string]toolHandler, error) {
	responseTools := make([]responses.ToolUnionParam, 0, len(tools))
	handlers := make(map[string]toolHandler, len(tools))

	for _, tool := range tools {
		if tool.Name == "" {
			return nil, nil, utils.WrapIfNotNil(errors.New("tool name is required"))
		}
		if tool.Handler == nil {
			return nil, nil, utils.WrapIfNotNil(fmt.Errorf("tool handler is required for %q", tool.Name))
		}
		if _, exists := handlers[tool.Name]; exists {
			return nil, nil, utils.WrapIfNotNil(fmt.Errorf("duplicate tool name %q", tool.Name))
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

func mapMCPTools(ctx context.Context, tools []model.MCPTool) ([]responses.ToolUnionParam, error) {
	responseTools := make([]responses.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		if tool.Name == "" {
			return nil, utils.WrapIfNotNil(errors.New("mcp tool name is required"))
		}
		if tool.URL == "" {
			return nil, utils.WrapIfNotNil(fmt.Errorf("mcp tool URL is required for %q", tool.Name))
		}

		authorization := extractAuthorization(tool.HTTPHeaders)
		allowedTools := append([]string(nil), tool.AllowedTools...)
		if len(allowedTools) == 0 {
			discoveredTools, err := mcp.FetchListOfTools(ctx, tool.URL, authorization)
			if err != nil {
				return nil, utils.WrapIfNotNil(
					fmt.Errorf("discover mcp tools for %q failed: %w", tool.Name, err),
				)
			}
			allowedTools = discoveredTools
		}

		param := responses.ToolMcpParam{
			ServerLabel: tool.Name,
			ServerURL:   openai.String(tool.URL),
			Headers:     copyHeaders(tool.HTTPHeaders),
			Type:        "mcp",
		}
		if len(allowedTools) > 0 {
			param.AllowedTools = responses.ToolMcpAllowedToolsUnionParam{
				OfMcpAllowedTools: append([]string(nil), allowedTools...),
			}
			param.RequireApproval = responses.ToolMcpRequireApprovalUnionParam{
				OfMcpToolApprovalFilter: &responses.ToolMcpRequireApprovalMcpToolApprovalFilterParam{
					Always: responses.ToolMcpRequireApprovalMcpToolApprovalFilterAlwaysParam{},
					Never: responses.ToolMcpRequireApprovalMcpToolApprovalFilterNeverParam{
						ToolNames: append([]string(nil), allowedTools...),
					},
				},
			}
		}

		responseTools = append(responseTools, responses.ToolUnionParam{
			OfMcp: &param,
		})
	}

	return responseTools, nil
}

func extractAuthorization(headers map[string]string) string {
	if len(headers) == 0 {
		return ""
	}

	authorization := ""
	for k, v := range headers {
		if strings.EqualFold(k, "Authorization") {
			authorization = v
		}
	}
	return authorization
}

func copyHeaders(headers map[string]string) map[string]string {
	if len(headers) == 0 {
		return nil
	}

	copied := make(map[string]string, len(headers))
	for k, v := range headers {
		copied[k] = v
	}
	return copied
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
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}

	var value T
	schema := reflector.Reflect(value)

	schemaJSON, err := json.Marshal(schema)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	var schemaMap map[string]any
	err = json.Unmarshal(schemaJSON, &schemaMap)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	return schemaMap, nil
}
