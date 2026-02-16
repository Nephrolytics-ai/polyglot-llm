package bedrock

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
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	bedrockdocument "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/invopop/jsonschema"
)

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
	log.Debugf("bedrock.structuredGenerator.AddPromptContext total_contexts=%d", len(g.promptContexts))
}

func (g *structuredGenerator[T]) AddPromptContextProvider(ctx context.Context, provider model.PromptContextProvider) {
	if provider == nil {
		return
	}

	g.promptContextMu.Lock()
	defer g.promptContextMu.Unlock()
	g.promptContextProviders = append(g.promptContextProviders, provider)
	logging.NewLogger(ctx).Debugf(
		"bedrock.structuredGenerator.AddPromptContextProvider total_providers=%d",
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
	log.Debugf("bedrock.textGenerator.AddPromptContext total_contexts=%d", len(g.promptContexts))
}

func (g *textGenerator) AddPromptContextProvider(ctx context.Context, provider model.PromptContextProvider) {
	if provider == nil {
		return
	}

	g.promptContextMu.Lock()
	defer g.promptContextMu.Unlock()
	g.promptContextProviders = append(g.promptContextProviders, provider)
	logging.NewLogger(ctx).Debugf(
		"bedrock.textGenerator.AddPromptContextProvider total_providers=%d",
		len(g.promptContextProviders),
	)
}

func (g *structuredGenerator[T]) Generate(ctx context.Context) (T, model.GenerationMetadata, error) {
	start := time.Now()
	modelName := resolveModelName(g.cfg)
	meta := initMetadata(modelName)
	defer setLatencyMetadata(meta, start)

	log := logging.NewLogger(ctx)
	system, messages, contextCount, err := g.messagesWithContext(ctx)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	schema, err := generateSchema[T]()
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}
	schemaJSON, err := json.Marshal(schema)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	messages[len(messages)-1].Content = []bedrocktypes.ContentBlock{
		&bedrocktypes.ContentBlockMemberText{
			Value: messages[len(messages)-1].Content[0].(*bedrocktypes.ContentBlockMemberText).Value +
				"\n\nReturn ONLY valid JSON that matches this schema:\n" + string(schemaJSON),
		},
	}

	allTools, cleanup, err := buildAllTools(ctx, g.cfg)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}
	defer cleanup()

	toolConfig, handlers, err := mapTools(allTools)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	client, err := newClient(ctx, g.cfg)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	log.Infof(
		"prompt=%q context_count=%d model=%q temperature=%v max_tokens=%v tools=%d mcp_tools=%d",
		g.prompt,
		contextCount,
		modelName,
		g.cfg.Temperature,
		g.cfg.MaxTokens,
		len(g.cfg.Tools),
		len(g.cfg.MCPTools),
	)

	inference := buildInferenceConfig(g.cfg)
	finalMessage, totals, stopReason, responseLatencyMs, err := runConverseFlow(
		ctx,
		client,
		modelName,
		system,
		messages,
		inference,
		toolConfig,
		handlers,
	)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}
	applyBedrockMetadata(meta, totals, stopReason, responseLatencyMs)

	text := strings.TrimSpace(extractTextFromMessage(finalMessage))
	if text == "" {
		err = errors.New("response output is empty")
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	payload := extractJSONPayload(text)
	var out T
	err = json.Unmarshal([]byte(payload), &out)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}
	return out, meta, nil
}

func (g *textGenerator) Generate(ctx context.Context) (string, model.GenerationMetadata, error) {
	start := time.Now()
	modelName := resolveModelName(g.cfg)
	meta := initMetadata(modelName)
	defer setLatencyMetadata(meta, start)

	log := logging.NewLogger(ctx)
	system, messages, contextCount, err := g.messagesWithContext(ctx)
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

	toolConfig, handlers, err := mapTools(allTools)
	if err != nil {
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}

	client, err := newClient(ctx, g.cfg)
	if err != nil {
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}

	log.Infof(
		"prompt=%q context_count=%d model=%q temperature=%v max_tokens=%v tools=%d mcp_tools=%d",
		g.prompt,
		contextCount,
		modelName,
		g.cfg.Temperature,
		g.cfg.MaxTokens,
		len(g.cfg.Tools),
		len(g.cfg.MCPTools),
	)

	inference := buildInferenceConfig(g.cfg)
	finalMessage, totals, stopReason, responseLatencyMs, err := runConverseFlow(
		ctx,
		client,
		modelName,
		system,
		messages,
		inference,
		toolConfig,
		handlers,
	)
	if err != nil {
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}
	applyBedrockMetadata(meta, totals, stopReason, responseLatencyMs)

	text := strings.TrimSpace(extractTextFromMessage(finalMessage))
	if text == "" {
		err = errors.New("response output is empty")
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}
	return text, meta, nil
}

func (g *structuredGenerator[T]) messagesWithContext(ctx context.Context) ([]bedrocktypes.SystemContentBlock, []bedrocktypes.Message, int, error) {
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

	return buildMessagesWithContext(g.prompt, contexts)
}

func (g *textGenerator) messagesWithContext(ctx context.Context) ([]bedrocktypes.SystemContentBlock, []bedrocktypes.Message, int, error) {
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

	return buildMessagesWithContext(g.prompt, contexts)
}

func buildMessagesWithContext(
	prompt string,
	contexts []*model.PromptContext,
) ([]bedrocktypes.SystemContentBlock, []bedrocktypes.Message, int, error) {
	system := make([]bedrocktypes.SystemContentBlock, 0)
	messages := make([]bedrocktypes.Message, 0, len(contexts)+1)
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
			system = append(system, &bedrocktypes.SystemContentBlockMemberText{Value: content})
		case model.ContextMessageTypeAssistant:
			messages = append(messages, bedrocktypes.Message{
				Role: bedrocktypes.ConversationRoleAssistant,
				Content: []bedrocktypes.ContentBlock{
					&bedrocktypes.ContentBlockMemberText{Value: content},
				},
			})
		case model.ContextMessageTypeHuman:
			messages = append(messages, bedrocktypes.Message{
				Role: bedrocktypes.ConversationRoleUser,
				Content: []bedrocktypes.ContentBlock{
					&bedrocktypes.ContentBlockMemberText{Value: content},
				},
			})
		default:
			messages = append(messages, bedrocktypes.Message{
				Role: bedrocktypes.ConversationRoleUser,
				Content: []bedrocktypes.ContentBlock{
					&bedrocktypes.ContentBlockMemberText{Value: content},
				},
			})
		}
	}

	messages = append(messages, bedrocktypes.Message{
		Role: bedrocktypes.ConversationRoleUser,
		Content: []bedrocktypes.ContentBlock{
			&bedrocktypes.ContentBlockMemberText{Value: prompt},
		},
	})

	return system, messages, contextCount, nil
}

func buildInferenceConfig(cfg model.GeneratorConfig) *bedrocktypes.InferenceConfiguration {
	if cfg.MaxTokens == nil && cfg.Temperature == nil {
		return nil
	}

	inference := &bedrocktypes.InferenceConfiguration{}
	if cfg.MaxTokens != nil {
		inference.MaxTokens = aws.Int32(int32(*cfg.MaxTokens))
	}
	if cfg.Temperature != nil {
		inference.Temperature = aws.Float32(float32(*cfg.Temperature))
	}
	return inference
}

func runConverseFlow(
	ctx context.Context,
	client *bedrockruntime.Client,
	modelID string,
	system []bedrocktypes.SystemContentBlock,
	initialMessages []bedrocktypes.Message,
	inference *bedrocktypes.InferenceConfiguration,
	toolConfig *bedrocktypes.ToolConfiguration,
	handlers map[string]toolHandler,
) (bedrocktypes.Message, flowUsageTotals, string, int64, error) {
	totals := flowUsageTotals{}
	history := append([]bedrocktypes.Message(nil), initialMessages...)
	var responseLatencyMs int64

	for round := 0; round < maxToolRounds; round++ {
		output, err := client.Converse(ctx, &bedrockruntime.ConverseInput{
			ModelId:         aws.String(modelID),
			Messages:        history,
			System:          system,
			InferenceConfig: inference,
			ToolConfig:      toolConfig,
		})
		if err != nil {
			return bedrocktypes.Message{}, totals, "", 0, utils.WrapIfNotNil(err)
		}

		totals.APICalls++
		if output.Usage != nil {
			totals.InputTokens += int64(aws.ToInt32(output.Usage.InputTokens))
			totals.OutputTokens += int64(aws.ToInt32(output.Usage.OutputTokens))
			totals.TotalTokens += int64(aws.ToInt32(output.Usage.TotalTokens))
			totals.CachedInputTokens += int64(aws.ToInt32(output.Usage.CacheReadInputTokens))
		}
		if output.Metrics != nil {
			responseLatencyMs += aws.ToInt64(output.Metrics.LatencyMs)
		}

		message, err := extractOutputMessage(output.Output)
		if err != nil {
			return bedrocktypes.Message{}, totals, "", responseLatencyMs, utils.WrapIfNotNil(err)
		}
		history = append(history, message)

		toolUses := extractToolUses(message)
		if len(toolUses) == 0 {
			return message, totals, string(output.StopReason), responseLatencyMs, nil
		}

		totals.ToolRounds = round + 1
		resultBlocks := make([]bedrocktypes.ContentBlock, 0, len(toolUses))
		for _, toolUse := range toolUses {
			name := strings.TrimSpace(aws.ToString(toolUse.Name))
			handler, ok := handlers[name]
			if !ok {
				return bedrocktypes.Message{}, totals, "", responseLatencyMs, utils.WrapIfNotNil(
					fmt.Errorf("no tool handler configured for function %q", name),
				)
			}

			argsBytes, marshalErr := toolUse.Input.MarshalSmithyDocument()
			if marshalErr != nil {
				return bedrocktypes.Message{}, totals, "", responseLatencyMs, utils.WrapIfNotNil(marshalErr)
			}

			result, callErr := handler(ctx, argsBytes)
			resultStatus := bedrocktypes.ToolResultStatusSuccess
			resultPayload := any(result)
			if callErr != nil {
				resultStatus = bedrocktypes.ToolResultStatusError
				resultPayload = map[string]any{"error": callErr.Error()}
			}

			resultBlocks = append(resultBlocks, &bedrocktypes.ContentBlockMemberToolResult{
				Value: bedrocktypes.ToolResultBlock{
					ToolUseId: toolUse.ToolUseId,
					Status:    resultStatus,
					Content: []bedrocktypes.ToolResultContentBlock{
						&bedrocktypes.ToolResultContentBlockMemberJson{
							Value: bedrockdocument.NewLazyDocument(resultPayload),
						},
					},
				},
			})
		}

		history = append(history, bedrocktypes.Message{
			Role:    bedrocktypes.ConversationRoleUser,
			Content: resultBlocks,
		})
	}

	return bedrocktypes.Message{}, totals, "", responseLatencyMs, utils.WrapIfNotNil(
		fmt.Errorf("exceeded tool call loop limit (%d)", maxToolRounds),
	)
}

func extractOutputMessage(output bedrocktypes.ConverseOutput) (bedrocktypes.Message, error) {
	if output == nil {
		return bedrocktypes.Message{}, utils.WrapIfNotNil(errors.New("converse output is nil"))
	}

	messageOutput, ok := output.(*bedrocktypes.ConverseOutputMemberMessage)
	if !ok || messageOutput == nil {
		return bedrocktypes.Message{}, utils.WrapIfNotNil(errors.New("converse output is not a message"))
	}
	return messageOutput.Value, nil
}

func extractToolUses(message bedrocktypes.Message) []bedrocktypes.ToolUseBlock {
	toolUses := make([]bedrocktypes.ToolUseBlock, 0)
	for _, block := range message.Content {
		toolUse, ok := block.(*bedrocktypes.ContentBlockMemberToolUse)
		if !ok || toolUse == nil {
			continue
		}
		toolUses = append(toolUses, toolUse.Value)
	}
	return toolUses
}

func extractTextFromMessage(message bedrocktypes.Message) string {
	parts := make([]string, 0)
	for _, block := range message.Content {
		textBlock, ok := block.(*bedrocktypes.ContentBlockMemberText)
		if !ok || textBlock == nil {
			continue
		}
		value := strings.TrimSpace(textBlock.Value)
		if value == "" {
			continue
		}
		parts = append(parts, value)
	}
	return strings.Join(parts, "\n")
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
