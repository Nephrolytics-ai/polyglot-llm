package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/logging"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
	"github.com/invopop/jsonschema"
	"google.golang.org/genai"
)

type toolHandler func(ctx context.Context, args json.RawMessage) (any, error)

type structuredGenerator[T any] struct {
	prompt                 string
	cfg                    model.GeneratorConfig
	promptContextMu        sync.RWMutex
	promptContexts         []*model.PromptContext
	promptContextProviders []model.PromptContextProvider
}

type textGenerator struct {
	prompt                 string
	cfg                    model.GeneratorConfig
	promptContextMu        sync.RWMutex
	promptContexts         []*model.PromptContext
	promptContextProviders []model.PromptContextProvider
}

func NewStructureContentGenerator[T any](prompt string, opts ...model.GeneratorOption) (model.ContentGenerator[T], error) {
	if strings.TrimSpace(prompt) == "" {
		return nil, utils.WrapIfNotNil(errors.New("prompt is required"))
	}

	cfg := model.ResolveGeneratorOpts(opts...)
	return &structuredGenerator[T]{
		prompt: prompt,
		cfg:    cfg,
	}, nil
}

func NewStringContentGenerator(prompt string, opts ...model.GeneratorOption) (model.ContentGenerator[string], error) {
	if strings.TrimSpace(prompt) == "" {
		return nil, utils.WrapIfNotNil(errors.New("prompt is required"))
	}

	cfg := model.ResolveGeneratorOpts(opts...)
	return &textGenerator{
		prompt: prompt,
		cfg:    cfg,
	}, nil
}

func (g *structuredGenerator[T]) AddPromptContext(ctx context.Context, messageType model.ContextMessageType, content string) {
	log := logging.NewLogger(ctx)
	g.promptContextMu.Lock()
	defer g.promptContextMu.Unlock()

	g.promptContexts = append(g.promptContexts, &model.PromptContext{
		MessageType: messageType,
		Content:     content,
	})
	log.Debugf("gemini.structuredGenerator.AddPromptContext total_contexts=%d", len(g.promptContexts))
}

func (g *structuredGenerator[T]) AddPromptContextProvider(ctx context.Context, provider model.PromptContextProvider) {
	if provider == nil {
		return
	}

	g.promptContextMu.Lock()
	defer g.promptContextMu.Unlock()
	g.promptContextProviders = append(g.promptContextProviders, provider)
	logging.NewLogger(ctx).Debugf(
		"gemini.structuredGenerator.AddPromptContextProvider total_providers=%d",
		len(g.promptContextProviders),
	)
}

func (g *textGenerator) AddPromptContext(ctx context.Context, messageType model.ContextMessageType, content string) {
	log := logging.NewLogger(ctx)
	g.promptContextMu.Lock()
	defer g.promptContextMu.Unlock()

	g.promptContexts = append(g.promptContexts, &model.PromptContext{
		MessageType: messageType,
		Content:     content,
	})
	log.Debugf("gemini.textGenerator.AddPromptContext total_contexts=%d", len(g.promptContexts))
}

func (g *textGenerator) AddPromptContextProvider(ctx context.Context, provider model.PromptContextProvider) {
	if provider == nil {
		return
	}

	g.promptContextMu.Lock()
	defer g.promptContextMu.Unlock()
	g.promptContextProviders = append(g.promptContextProviders, provider)
	logging.NewLogger(ctx).Debugf(
		"gemini.textGenerator.AddPromptContextProvider total_providers=%d",
		len(g.promptContextProviders),
	)
}

func (g *structuredGenerator[T]) Generate(ctx context.Context) (T, model.GenerationMetadata, error) {
	start := time.Now()
	modelName := resolveGenerationModelName(g.cfg)
	meta := initMetadata(modelName)
	defer setLatencyMetadata(meta, start)

	log := logging.NewLogger(ctx)
	systemInstruction, contents, contextCount, err := g.contentsWithContext(ctx)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	allTools, cleanup, err := buildAllTools(ctx, g.cfg)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}
	defer cleanup()

	genTools, handlers, err := mapTools(allTools)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	config := buildGenerateContentConfig(g.cfg, systemInstruction, genTools)
	schema, err := generateJSONSchema[T]()
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}
	config.ResponseMIMEType = "application/json"
	config.ResponseJsonSchema = schema

	client, err := newAPIClient(ctx, g.cfg)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	log.Infof(
		"prompt=%q context_count=%d model=%q temperature=%v max_tokens=%v reasoning=%v tools=%d mcp_tools=%d",
		g.prompt,
		contextCount,
		modelName,
		g.cfg.Temperature,
		g.cfg.MaxTokens,
		g.cfg.ReasoningLevel,
		len(g.cfg.Tools),
		len(g.cfg.MCPTools),
	)

	response, totals, err := runGenerateFlow(ctx, client, modelName, contents, config, handlers)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	applyGenerateMetadata(meta, response, totals)
	text := strings.TrimSpace(response.Text())
	if text == "" {
		err = errors.New("response output is empty")
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	var out T
	err = json.Unmarshal([]byte(text), &out)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}
	return out, meta, nil
}

func (g *textGenerator) Generate(ctx context.Context) (string, model.GenerationMetadata, error) {
	start := time.Now()
	modelName := resolveGenerationModelName(g.cfg)
	meta := initMetadata(modelName)
	defer setLatencyMetadata(meta, start)

	log := logging.NewLogger(ctx)
	systemInstruction, contents, contextCount, err := g.contentsWithContext(ctx)
	if err != nil {
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}

	allTools, cleanup, err := buildAllTools(ctx, g.cfg)
	if err != nil {
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}
	defer cleanup()

	genTools, handlers, err := mapTools(allTools)
	if err != nil {
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}

	config := buildGenerateContentConfig(g.cfg, systemInstruction, genTools)
	client, err := newAPIClient(ctx, g.cfg)
	if err != nil {
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}

	log.Infof(
		"prompt=%q context_count=%d model=%q temperature=%v max_tokens=%v reasoning=%v tools=%d mcp_tools=%d",
		g.prompt,
		contextCount,
		modelName,
		g.cfg.Temperature,
		g.cfg.MaxTokens,
		g.cfg.ReasoningLevel,
		len(g.cfg.Tools),
		len(g.cfg.MCPTools),
	)

	response, totals, err := runGenerateFlow(ctx, client, modelName, contents, config, handlers)
	if err != nil {
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}
	applyGenerateMetadata(meta, response, totals)

	text := strings.TrimSpace(response.Text())
	if text == "" {
		err = errors.New("response output is empty")
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}
	return text, meta, nil
}

func (g *structuredGenerator[T]) contentsWithContext(ctx context.Context) (*genai.Content, []*genai.Content, int, error) {
	g.promptContextMu.RLock()
	contexts := append([]*model.PromptContext(nil), g.promptContexts...)
	providers := append([]model.PromptContextProvider(nil), g.promptContextProviders...)
	g.promptContextMu.RUnlock()

	for _, provider := range providers {
		provided, err := provider.GenerateContext(ctx)
		if err != nil {
			return nil, nil, 0, utils.WrapIfNotNil(err)
		}
		contexts = append(contexts, provided...)
	}

	return buildContentsWithContext(g.prompt, contexts)
}

func (g *textGenerator) contentsWithContext(ctx context.Context) (*genai.Content, []*genai.Content, int, error) {
	g.promptContextMu.RLock()
	contexts := append([]*model.PromptContext(nil), g.promptContexts...)
	providers := append([]model.PromptContextProvider(nil), g.promptContextProviders...)
	g.promptContextMu.RUnlock()

	for _, provider := range providers {
		provided, err := provider.GenerateContext(ctx)
		if err != nil {
			return nil, nil, 0, utils.WrapIfNotNil(err)
		}
		contexts = append(contexts, provided...)
	}

	return buildContentsWithContext(g.prompt, contexts)
}

func buildContentsWithContext(prompt string, contexts []*model.PromptContext) (*genai.Content, []*genai.Content, int, error) {
	systemParts := make([]string, 0)
	contents := make([]*genai.Content, 0, len(contexts)+1)
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
		switch contextItem.MessageType {
		case model.ContextMessageTypeSystem:
			systemParts = append(systemParts, content)
		case model.ContextMessageTypeAssistant:
			contents = append(contents, genai.NewContentFromText(content, genai.RoleModel))
		case model.ContextMessageTypeHuman:
			contents = append(contents, genai.NewContentFromText(content, genai.RoleUser))
		default:
			contents = append(contents, genai.NewContentFromText(content, genai.RoleUser))
		}
	}

	contents = append(contents, genai.NewContentFromText(prompt, genai.RoleUser))

	if len(systemParts) == 0 {
		return nil, contents, contextCount, nil
	}

	systemInstruction := genai.NewContentFromText(strings.Join(systemParts, "\n\n"), genai.RoleUser)
	return systemInstruction, contents, contextCount, nil
}

func buildGenerateContentConfig(
	cfg model.GeneratorConfig,
	systemInstruction *genai.Content,
	tools []*genai.Tool,
) *genai.GenerateContentConfig {
	config := &genai.GenerateContentConfig{}

	if systemInstruction != nil {
		config.SystemInstruction = systemInstruction
	}
	if cfg.Temperature != nil {
		temp := float32(*cfg.Temperature)
		config.Temperature = &temp
	}
	if cfg.MaxTokens != nil {
		config.MaxOutputTokens = int32(*cfg.MaxTokens)
	}
	if cfg.ReasoningLevel != nil {
		config.ThinkingConfig = &genai.ThinkingConfig{
			ThinkingLevel: mapReasoningLevel(*cfg.ReasoningLevel),
		}
	}
	if len(tools) > 0 {
		config.Tools = tools
		config.ToolConfig = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeAuto,
			},
		}
	}

	return config
}

func mapReasoningLevel(level model.ReasoningLevel) genai.ThinkingLevel {
	switch level {
	case model.ReasoningLevelNone:
		return genai.ThinkingLevelMinimal
	case model.ReasoningLevelLow:
		return genai.ThinkingLevelLow
	case model.ReasoningLevelMed:
		return genai.ThinkingLevelMedium
	case model.ReasoningLevelHigh:
		return genai.ThinkingLevelHigh
	default:
		return genai.ThinkingLevelMedium
	}
}

func runGenerateFlow(
	ctx context.Context,
	client *genai.Client,
	modelName string,
	initialContents []*genai.Content,
	config *genai.GenerateContentConfig,
	handlers map[string]toolHandler,
) (*genai.GenerateContentResponse, generationTotals, error) {
	totals := generationTotals{}
	history := append([]*genai.Content(nil), initialContents...)

	response, configToUse, err := generateWithThinkingFallback(ctx, client, modelName, history, config)
	if err != nil {
		return nil, totals, utils.WrapIfNotNil(err)
	}
	accumulateGenerationTotals(&totals, response)

	for round := 0; round < maxToolRounds; round++ {
		functionCalls := response.FunctionCalls()
		if len(functionCalls) == 0 {
			return response, totals, nil
		}
		totals.ToolRounds = round + 1

		for _, call := range functionCalls {
			handler, ok := handlers[call.Name]
			if !ok {
				return nil, totals, utils.WrapIfNotNil(
					fmt.Errorf("no tool handler configured for function %q", call.Name),
				)
			}

			argsBytes, marshalErr := json.Marshal(call.Args)
			if marshalErr != nil {
				return nil, totals, utils.WrapIfNotNil(marshalErr)
			}

			result, callErr := handler(ctx, argsBytes)
			if callErr != nil {
				return nil, totals, utils.WrapIfNotNil(callErr)
			}

			history = append(history, genai.NewContentFromFunctionCall(call.Name, call.Args, genai.RoleModel))

			toolOutput := map[string]any{"output": result}
			if strings.TrimSpace(call.ID) != "" {
				toolOutput["id"] = call.ID
			}
			history = append(history, genai.NewContentFromFunctionResponse(call.Name, toolOutput, genai.RoleUser))
		}

		response, _, err = generateWithThinkingFallback(ctx, client, modelName, history, configToUse)
		if err != nil {
			return nil, totals, utils.WrapIfNotNil(err)
		}
		accumulateGenerationTotals(&totals, response)
	}

	return nil, totals, utils.WrapIfNotNil(fmt.Errorf("exceeded tool call loop limit (%d)", maxToolRounds))
}

func generateWithThinkingFallback(
	ctx context.Context,
	client *genai.Client,
	modelName string,
	contents []*genai.Content,
	config *genai.GenerateContentConfig,
) (*genai.GenerateContentResponse, *genai.GenerateContentConfig, error) {
	response, err := client.Models.GenerateContent(ctx, modelName, contents, config)
	if err == nil {
		return response, config, nil
	}

	if config == nil || config.ThinkingConfig == nil || !utils.ContainsErrorSubstring(err, "Thinking level is not supported for this model") {
		return nil, config, utils.WrapIfNotNil(err)
	}

	logging.NewLogger(ctx).Warnf(
		"thinking level unsupported for model %q; retrying without thinking config",
		modelName,
	)

	fallback := *config
	fallback.ThinkingConfig = nil

	response, err = client.Models.GenerateContent(ctx, modelName, contents, &fallback)
	if err != nil {
		return nil, &fallback, utils.WrapIfNotNil(err)
	}

	return response, &fallback, nil
}

func mapTools(tools []model.Tool) ([]*genai.Tool, map[string]toolHandler, error) {
	if len(tools) == 0 {
		return nil, nil, nil
	}

	declarations := make([]*genai.FunctionDeclaration, 0, len(tools))
	handlers := make(map[string]toolHandler, len(tools))

	for _, tool := range tools {
		if strings.TrimSpace(tool.Name) == "" {
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

		declarations = append(declarations, &genai.FunctionDeclaration{
			Name:                 tool.Name,
			Description:          tool.Description,
			ParametersJsonSchema: parameters,
		})
		handlers[tool.Name] = tool.Handler
	}

	return []*genai.Tool{
		{
			FunctionDeclarations: declarations,
		},
	}, handlers, nil
}

func generateJSONSchema[T any]() (map[string]any, error) {
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
