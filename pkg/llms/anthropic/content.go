package anthropic

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
	log.Debugf("anthropic.structuredGenerator.AddPromptContext total_contexts=%d", len(g.promptContexts))
}

func (g *structuredGenerator[T]) AddPromptContextProvider(ctx context.Context, provider model.PromptContextProvider) {
	if provider == nil {
		return
	}

	g.promptContextMu.Lock()
	defer g.promptContextMu.Unlock()
	g.promptContextProviders = append(g.promptContextProviders, provider)
	logging.NewLogger(ctx).Debugf(
		"anthropic.structuredGenerator.AddPromptContextProvider total_providers=%d",
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
	log.Debugf("anthropic.textGenerator.AddPromptContext total_contexts=%d", len(g.promptContexts))
}

func (g *textGenerator) AddPromptContextProvider(ctx context.Context, provider model.PromptContextProvider) {
	if provider == nil {
		return
	}

	g.promptContextMu.Lock()
	defer g.promptContextMu.Unlock()
	g.promptContextProviders = append(g.promptContextProviders, provider)
	logging.NewLogger(ctx).Debugf(
		"anthropic.textGenerator.AddPromptContextProvider total_providers=%d",
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

	system, messages, contextCount, err := g.messagesWithContext(ctx, schemaInstruction)
	if err != nil {
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	tools, handlers, mcpServers, cleanup, err := buildAllTools(ctx, cfg)
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

	response, totals, err := runMessageFlow(ctx, g.client, cfg, modelName, system, messages, tools, handlers, mcpServers)
	if err != nil {
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}
	applyAnthropicMetadata(meta, response, totals)

	text := strings.TrimSpace(extractTextFromContentBlocks(response.Content))
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

	system, messages, contextCount, err := g.messagesWithContext(ctx, "")
	if err != nil {
		return "", meta, utils.WrapIfNotNil(err)
	}

	tools, handlers, mcpServers, cleanup, err := buildAllTools(ctx, cfg)
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

	response, totals, err := runMessageFlow(ctx, g.client, cfg, modelName, system, messages, tools, handlers, mcpServers)
	if err != nil {
		return "", meta, utils.WrapIfNotNil(err)
	}
	applyAnthropicMetadata(meta, response, totals)

	text := strings.TrimSpace(extractTextFromContentBlocks(response.Content))
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
	system string,
	initialMessages []anthropicMessage,
	tools []anthropicTool,
	handlers map[string]toolHandler,
	mcpServers []anthropicMCPServer,
) (*anthropicMessageResponse, flowUsageTotals, error) {
	log := logging.NewLogger(ctx)
	totals := flowUsageTotals{}
	messages := append([]anthropicMessage(nil), initialMessages...)

	for round := 0; round < maxToolRounds; round++ {
		request := anthropicMessageRequest{
			Model:      modelName,
			MaxTokens:  resolveMaxTokens(cfg),
			System:     strings.TrimSpace(system),
			Messages:   append([]anthropicMessage(nil), messages...),
			Tools:      append([]anthropicTool(nil), tools...),
			MCPServers: append([]anthropicMCPServer(nil), mcpServers...),
		}
		if cfg.Temperature != nil {
			request.Temperature = cfg.Temperature
		}

		response, err := client.createMessage(ctx, request, len(mcpServers) > 0)
		if err != nil {
			return nil, totals, utils.WrapIfNotNil(err)
		}
		if response == nil {
			return nil, totals, utils.WrapIfNotNil(errors.New("anthropic API returned nil response"))
		}

		accumulateUsageTotals(&totals, response)
		messages = append(messages, anthropicMessage{
			Role:    "assistant",
			Content: append([]anthropicContentBlock(nil), response.Content...),
		})

		results := make([]anthropicContentBlock, 0)
		localToolCalls := 0
		for _, block := range response.Content {
			if block.Type != "tool_use" {
				continue
			}

			handler, found := handlers[block.Name]
			if !found {
				log.Warnf("tool_use for %q has no local handler; assuming remote MCP handling", block.Name)
				continue
			}

			localToolCalls++
			result, callErr := handler(ctx, block.Input)
			if callErr != nil {
				return nil, totals, utils.WrapIfNotNil(callErr)
			}

			resultJSON, marshalErr := json.Marshal(result)
			if marshalErr != nil {
				return nil, totals, utils.WrapIfNotNil(marshalErr)
			}
			resultJSONText, marshalTextErr := json.Marshal(string(resultJSON))
			if marshalTextErr != nil {
				return nil, totals, utils.WrapIfNotNil(marshalTextErr)
			}

			results = append(results, anthropicContentBlock{
				Type:      "tool_result",
				ToolUseID: block.ID,
				Content:   resultJSONText,
			})
		}

		if localToolCalls == 0 {
			return response, totals, nil
		}

		totals.ToolRounds = round + 1
		messages = append(messages, anthropicMessage{Role: "user", Content: results})
	}

	return nil, totals, utils.WrapIfNotNil(fmt.Errorf("exceeded tool call loop limit (%d)", maxToolRounds))
}

func (g *structuredGenerator[T]) messagesWithContext(
	ctx context.Context,
	promptSuffix string,
) (string, []anthropicMessage, int, error) {
	g.promptContextMu.RLock()
	contexts := append([]*model.PromptContext(nil), g.promptContexts...)
	providers := append([]model.PromptContextProvider(nil), g.promptContextProviders...)
	g.promptContextMu.RUnlock()

	for _, provider := range providers {
		provided, err := provider.GenerateContext(ctx)
		if err != nil {
			return "", nil, 0, utils.WrapIfNotNil(err)
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
) (string, []anthropicMessage, int, error) {
	g.promptContextMu.RLock()
	contexts := append([]*model.PromptContext(nil), g.promptContexts...)
	providers := append([]model.PromptContextProvider(nil), g.promptContextProviders...)
	g.promptContextMu.RUnlock()

	for _, provider := range providers {
		provided, err := provider.GenerateContext(ctx)
		if err != nil {
			return "", nil, 0, utils.WrapIfNotNil(err)
		}
		contexts = append(contexts, provided...)
	}

	prompt := g.prompt
	if strings.TrimSpace(promptSuffix) != "" {
		prompt += "\n\n" + promptSuffix
	}
	return buildMessagesWithContext(prompt, contexts)
}

func buildMessagesWithContext(prompt string, contexts []*model.PromptContext) (string, []anthropicMessage, int, error) {
	systemParts := make([]string, 0)
	messages := make([]anthropicMessage, 0, len(contexts)+1)
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
			messages = append(messages, makeTextMessage("assistant", content))
		case model.ContextMessageTypeHuman:
			messages = append(messages, makeTextMessage("user", content))
		default:
			messages = append(messages, makeTextMessage("user", content))
		}
	}

	messages = append(messages, makeTextMessage("user", prompt))
	return strings.Join(systemParts, "\n\n"), messages, contextCount, nil
}

func makeTextMessage(role string, content string) anthropicMessage {
	return anthropicMessage{
		Role: role,
		Content: []anthropicContentBlock{
			{
				Type: "text",
				Text: content,
			},
		},
	}
}

func extractTextFromContentBlocks(content []anthropicContentBlock) string {
	if len(content) == 0 {
		return ""
	}

	parts := make([]string, 0, len(content))
	for _, block := range content {
		if block.Type != "text" {
			continue
		}
		trimmed := strings.TrimSpace(block.Text)
		if trimmed == "" {
			continue
		}
		parts = append(parts, trimmed)
	}
	return strings.Join(parts, "\n")
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
