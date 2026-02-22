package huggingface

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
)

type structuredGenerator[T any] struct {
	client                 *apiClient
	prompt                 string
	cfg                    model.GeneratorConfig
	promptContextMu        sync.RWMutex
	promptContexts         []*model.PromptContext
	promptContextProviders []model.PromptContextProvider
}

type textGenerator struct {
	client                 *apiClient
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
	client, err := newAPIClient(cfg)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	return &structuredGenerator[T]{
		client: client,
		prompt: prompt,
		cfg:    cfg,
	}, nil
}

func NewStringContentGenerator(prompt string, opts ...model.GeneratorOption) (model.ContentGenerator[string], error) {
	if strings.TrimSpace(prompt) == "" {
		return nil, utils.WrapIfNotNil(errors.New("prompt is required"))
	}

	cfg := model.ResolveGeneratorOpts(opts...)
	client, err := newAPIClient(cfg)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	return &textGenerator{
		client: client,
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
	log.Debugf("huggingface.structuredGenerator.AddPromptContext total_contexts=%d", len(g.promptContexts))
}

func (g *structuredGenerator[T]) AddPromptContextProvider(ctx context.Context, provider model.PromptContextProvider) {
	if provider == nil {
		return
	}

	g.promptContextMu.Lock()
	defer g.promptContextMu.Unlock()
	g.promptContextProviders = append(g.promptContextProviders, provider)
	logging.NewLogger(ctx).Debugf(
		"huggingface.structuredGenerator.AddPromptContextProvider total_providers=%d",
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
	log.Debugf("huggingface.textGenerator.AddPromptContext total_contexts=%d", len(g.promptContexts))
}

func (g *textGenerator) AddPromptContextProvider(ctx context.Context, provider model.PromptContextProvider) {
	if provider == nil {
		return
	}

	g.promptContextMu.Lock()
	defer g.promptContextMu.Unlock()
	g.promptContextProviders = append(g.promptContextProviders, provider)
	logging.NewLogger(ctx).Debugf(
		"huggingface.textGenerator.AddPromptContextProvider total_providers=%d",
		len(g.promptContextProviders),
	)
}

func (g *structuredGenerator[T]) Generate(ctx context.Context) (T, model.GenerationMetadata, error) {
	start := time.Now()
	log := logging.NewLogger(ctx)

	cfg, err := normalizeGeneratorOptionsForProvider(g.cfg, log)
	if err != nil {
		var zero T
		return zero, nil, utils.WrapIfNotNil(err)
	}

	modelName := resolveModelName(cfg)
	meta := initMetadata(modelName)
	defer setLatencyMetadata(meta, start)

	schema, err := generateJSONSchema[T]()
	if err != nil {
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}
	schemaInstruction, err := buildStructuredOutputInstruction(schema)
	if err != nil {
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	messages, contextCount, err := g.messagesWithContext(ctx, schemaInstruction)
	if err != nil {
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	tools, handlers, cleanup, err := buildAllTools(ctx, cfg)
	if err != nil {
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}
	defer cleanup()

	log.Infof(
		"prompt=%q context_count=%d model=%q temperature=%v max_tokens=%v tools=%d mcp_tools=%d",
		g.prompt,
		contextCount,
		modelName,
		cfg.Temperature,
		cfg.MaxTokens,
		len(cfg.Tools),
		len(cfg.MCPTools),
	)

	response, totals, err := runMessageFlow(ctx, g.client, cfg, modelName, messages, tools, handlers)
	if err != nil {
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}
	applyHuggingFaceMetadata(meta, response, totals)

	text := extractTextFromResponse(response)
	if text == "" {
		err = errors.New("response output is empty")
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	var out T
	err = json.Unmarshal([]byte(extractJSONPayload(text)), &out)
	if err != nil {
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}
	return out, meta, nil
}

func (g *textGenerator) Generate(ctx context.Context) (string, model.GenerationMetadata, error) {
	start := time.Now()
	log := logging.NewLogger(ctx)

	cfg, err := normalizeGeneratorOptionsForProvider(g.cfg, log)
	if err != nil {
		return "", nil, utils.WrapIfNotNil(err)
	}

	modelName := resolveModelName(cfg)
	meta := initMetadata(modelName)
	defer setLatencyMetadata(meta, start)

	messages, contextCount, err := g.messagesWithContext(ctx, "")
	if err != nil {
		return "", meta, utils.WrapIfNotNil(err)
	}

	tools, handlers, cleanup, err := buildAllTools(ctx, cfg)
	if err != nil {
		return "", meta, utils.WrapIfNotNil(err)
	}
	defer cleanup()

	log.Infof(
		"prompt=%q context_count=%d model=%q temperature=%v max_tokens=%v tools=%d mcp_tools=%d",
		g.prompt,
		contextCount,
		modelName,
		cfg.Temperature,
		cfg.MaxTokens,
		len(cfg.Tools),
		len(cfg.MCPTools),
	)

	response, totals, err := runMessageFlow(ctx, g.client, cfg, modelName, messages, tools, handlers)
	if err != nil {
		return "", meta, utils.WrapIfNotNil(err)
	}
	applyHuggingFaceMetadata(meta, response, totals)

	text := extractTextFromResponse(response)
	if text == "" {
		err = errors.New("response output is empty")
		return "", meta, utils.WrapIfNotNil(err)
	}

	return text, meta, nil
}

func runMessageFlow(
	ctx context.Context,
	client *apiClient,
	cfg model.GeneratorConfig,
	modelName string,
	initialMessages []chatMessage,
	tools []chatTool,
	handlers map[string]toolHandler,
) (*chatCompletionResponse, flowUsageTotals, error) {
	log := logging.NewLogger(ctx)
	totals := flowUsageTotals{}
	messages := append([]chatMessage(nil), initialMessages...)

	for round := 0; round < maxToolRounds; round++ {
		request := chatCompletionRequest{
			Model:    modelName,
			Messages: append([]chatMessage(nil), messages...),
		}
		request.MaxTokens = resolveMaxTokens(cfg)
		if cfg.Temperature != nil {
			request.Temperature = cfg.Temperature
		}
		if len(tools) > 0 {
			request.Tools = append([]chatTool(nil), tools...)
		}

		response, err := client.createChatCompletion(ctx, request)
		if err != nil {
			return nil, totals, utils.WrapIfNotNil(err)
		}
		if response == nil {
			return nil, totals, utils.WrapIfNotNil(errors.New("huggingface API returned nil response"))
		}

		accumulateUsageTotals(&totals, response)

		if len(response.Choices) == 0 {
			return nil, totals, utils.WrapIfNotNil(errors.New("huggingface API returned no choices"))
		}

		assistantMsg := response.Choices[0].Message
		messages = append(messages, assistantMsg)

		if len(assistantMsg.ToolCalls) == 0 {
			return response, totals, nil
		}

		localToolCalls := 0
		for _, toolCall := range assistantMsg.ToolCalls {
			handler, found := handlers[toolCall.Function.Name]
			if !found {
				log.Warnf("tool_call for %q has no handler; skipping", toolCall.Function.Name)
				continue
			}

			localToolCalls++
			result, callErr := handler(ctx, json.RawMessage(toolCall.Function.Arguments))
			if callErr != nil {
				return nil, totals, utils.WrapIfNotNil(callErr)
			}

			resultJSON, marshalErr := json.Marshal(result)
			if marshalErr != nil {
				return nil, totals, utils.WrapIfNotNil(marshalErr)
			}

			messages = append(messages, chatMessage{
				Role:       "tool",
				Content:    string(resultJSON),
				ToolCallID: toolCall.ID,
			})
		}

		if localToolCalls == 0 {
			return response, totals, nil
		}

		totals.ToolRounds = round + 1
	}

	return nil, totals, utils.WrapIfNotNil(fmt.Errorf("exceeded tool call loop limit (%d)", maxToolRounds))
}

func (g *structuredGenerator[T]) messagesWithContext(
	ctx context.Context,
	promptSuffix string,
) ([]chatMessage, int, error) {
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

	prompt := g.prompt
	if strings.TrimSpace(promptSuffix) != "" {
		prompt += "\n\n" + promptSuffix
	}
	return buildMessagesWithContext(prompt, contexts)
}

func (g *textGenerator) messagesWithContext(
	ctx context.Context,
	promptSuffix string,
) ([]chatMessage, int, error) {
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

	prompt := g.prompt
	if strings.TrimSpace(promptSuffix) != "" {
		prompt += "\n\n" + promptSuffix
	}
	return buildMessagesWithContext(prompt, contexts)
}

func buildMessagesWithContext(prompt string, contexts []*model.PromptContext) ([]chatMessage, int, error) {
	messages := make([]chatMessage, 0, len(contexts)+1)
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
			messages = append(messages, chatMessage{Role: "system", Content: content})
		case model.ContextMessageTypeAssistant:
			messages = append(messages, chatMessage{Role: "assistant", Content: content})
		case model.ContextMessageTypeHuman:
			messages = append(messages, chatMessage{Role: "user", Content: content})
		default:
			messages = append(messages, chatMessage{Role: "user", Content: content})
		}
	}

	messages = append(messages, chatMessage{Role: "user", Content: prompt})
	return messages, contextCount, nil
}

func extractTextFromResponse(response *chatCompletionResponse) string {
	if response == nil || len(response.Choices) == 0 {
		return ""
	}
	return strings.TrimSpace(response.Choices[0].Message.Content)
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

func buildStructuredOutputInstruction(schema map[string]any) (string, error) {
	schemaBytes, err := json.Marshal(schema)
	if err != nil {
		return "", utils.WrapIfNotNil(err)
	}

	return "Return ONLY valid JSON matching this schema. Do not include markdown fences.\n" + string(schemaBytes), nil
}

func extractJSONPayload(text string) string {
	trimmed := strings.TrimSpace(text)
	trimmed = strings.TrimPrefix(trimmed, "```json")
	trimmed = strings.TrimPrefix(trimmed, "```")
	trimmed = strings.TrimSuffix(trimmed, "```")
	trimmed = strings.TrimSpace(trimmed)

	start := strings.Index(trimmed, "{")
	end := strings.LastIndex(trimmed, "}")
	if start >= 0 && end > start {
		return strings.TrimSpace(trimmed[start : end+1])
	}
	return trimmed
}
