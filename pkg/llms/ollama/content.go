package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/logging"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
	"github.com/invopop/jsonschema"
	ollamasdk "github.com/rozoomcool/go-ollama-sdk"
)

type toolHandler func(ctx context.Context, args json.RawMessage) (any, error)

type structuredGenerator[T any] struct {
	client                 *client
	prompt                 string
	cfg                    model.GeneratorConfig
	promptContextMu        sync.RWMutex
	promptContexts         []*model.PromptContext
	promptContextProviders []model.PromptContextProvider
}

type textGenerator struct {
	client                 *client
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
	c := newClient(cfg)
	return &structuredGenerator[T]{
		client: c,
		prompt: prompt,
		cfg:    cfg,
	}, nil
}

func NewStringContentGenerator(prompt string, opts ...model.GeneratorOption) (model.ContentGenerator[string], error) {
	if strings.TrimSpace(prompt) == "" {
		return nil, utils.WrapIfNotNil(errors.New("prompt is required"))
	}

	cfg := model.ResolveGeneratorOpts(opts...)
	c := newClient(cfg)
	return &textGenerator{
		client: c,
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
	log.Debugf("ollama.structuredGenerator.AddPromptContext total_contexts=%d", len(g.promptContexts))
}

func (g *structuredGenerator[T]) AddPromptContextProvider(ctx context.Context, provider model.PromptContextProvider) {
	if provider == nil {
		return
	}

	g.promptContextMu.Lock()
	defer g.promptContextMu.Unlock()
	g.promptContextProviders = append(g.promptContextProviders, provider)
	logging.NewLogger(ctx).Debugf(
		"ollama.structuredGenerator.AddPromptContextProvider total_providers=%d",
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
	log.Debugf("ollama.textGenerator.AddPromptContext total_contexts=%d", len(g.promptContexts))
}

func (g *textGenerator) AddPromptContextProvider(ctx context.Context, provider model.PromptContextProvider) {
	if provider == nil {
		return
	}

	g.promptContextMu.Lock()
	defer g.promptContextMu.Unlock()
	g.promptContextProviders = append(g.promptContextProviders, provider)
	logging.NewLogger(ctx).Debugf(
		"ollama.textGenerator.AddPromptContextProvider total_providers=%d",
		len(g.promptContextProviders),
	)
}

func (g *structuredGenerator[T]) Generate(ctx context.Context) (T, model.GenerationMetadata, error) {
	start := time.Now()
	modelName := resolveGenerationModelName(g.cfg)
	meta := initMetadata(modelName)
	defer setLatencyMetadata(meta, start)

	log := logging.NewLogger(ctx)
	messages, contextCount, err := g.messagesWithContext(ctx)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	schema, err := generateJSONSchema[T]()
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}
	schemaInstruction, err := buildStructuredOutputInstruction(schema)
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

	modelTools, handlers, err := mapTools(allTools)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	messages = append(messages, ollamasdk.ChatMessage{
		Role:    "user",
		Content: schemaInstruction,
	})

	log.Infof(
		"prompt=%q context_count=%d model=%q tools=%d mcp_tools=%d base_url=%q",
		g.prompt,
		contextCount,
		modelName,
		len(g.cfg.Tools),
		len(g.cfg.MCPTools),
		g.client.baseURL,
	)

	finalText, totals, err := runChatFlow(ctx, g.client, modelName, g.cfg, messages, modelTools, handlers)
	if err != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}
	applyOllamaMetadata(meta, totals)

	payload := extractJSONPayload(finalText)
	var out T
	err = json.Unmarshal([]byte(payload), &out)
	if err == nil {
		return out, meta, nil
	}

	// Ollama may return explanatory text after tool calls; do one repair round to force valid JSON.
	log.Warnf("structured output parse failed, attempting repair: %v", err)
	repaired, repairErr := g.repairStructuredJSON(ctx, modelName, schema, finalText)
	if repairErr != nil {
		log.Errorf("error: %v", err)
		var zero T
		return zero, meta, utils.WrapIfNotNil(err)
	}

	err = json.Unmarshal([]byte(extractJSONPayload(repaired)), &out)
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
	messages, contextCount, err := g.messagesWithContext(ctx)
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

	modelTools, handlers, err := mapTools(allTools)
	if err != nil {
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}

	log.Infof(
		"prompt=%q context_count=%d model=%q tools=%d mcp_tools=%d base_url=%q",
		g.prompt,
		contextCount,
		modelName,
		len(g.cfg.Tools),
		len(g.cfg.MCPTools),
		g.client.baseURL,
	)

	finalText, totals, err := runChatFlow(ctx, g.client, modelName, g.cfg, messages, modelTools, handlers)
	if err != nil {
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}
	applyOllamaMetadata(meta, totals)

	finalText = strings.TrimSpace(finalText)
	if finalText == "" {
		err = errors.New("response output is empty")
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}
	return finalText, meta, nil
}

func (g *structuredGenerator[T]) messagesWithContext(ctx context.Context) ([]ollamasdk.ChatMessage, int, error) {
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

	return buildMessagesWithContext(g.prompt, contexts)
}

func (g *textGenerator) messagesWithContext(ctx context.Context) ([]ollamasdk.ChatMessage, int, error) {
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

	return buildMessagesWithContext(g.prompt, contexts)
}

func buildMessagesWithContext(prompt string, contexts []*model.PromptContext) ([]ollamasdk.ChatMessage, int, error) {
	messages := make([]ollamasdk.ChatMessage, 0, len(contexts)+1)
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
		role := "user"
		switch contextItem.MessageType {
		case model.ContextMessageTypeSystem:
			role = "system"
		case model.ContextMessageTypeAssistant:
			role = "assistant"
		case model.ContextMessageTypeHuman:
			role = "user"
		default:
			role = "user"
		}

		messages = append(messages, ollamasdk.ChatMessage{
			Role:    role,
			Content: content,
		})
	}

	messages = append(messages, ollamasdk.ChatMessage{
		Role:    "user",
		Content: prompt,
	})

	return messages, contextCount, nil
}

type flowUsageTotals struct {
	APICalls     int
	ToolRounds   int
	InputTokens  int64
	OutputTokens int64
	TotalTokens  int64
}

type ollamaChatRequest struct {
	Model    string              `json:"model"`
	Messages []ollamaChatMessage `json:"messages"`
	Stream   bool                `json:"stream"`
	Tools    []ollamaToolDef     `json:"tools,omitempty"`
	Options  *ollamaChatOptions  `json:"options,omitempty"`
}

type ollamaChatResponse struct {
	Model           string            `json:"model"`
	Message         ollamaChatMessage `json:"message"`
	Done            bool              `json:"done"`
	PromptEvalCount int64             `json:"prompt_eval_count,omitempty"`
	EvalCount       int64             `json:"eval_count,omitempty"`
	Error           string            `json:"error,omitempty"`
}

type ollamaErrorResponse struct {
	Error string `json:"error"`
}

type ollamaChatMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content,omitempty"`
	ToolCalls  []ollamaToolCall `json:"tool_calls,omitempty"`
	Name       string           `json:"name,omitempty"`
	ToolName   string           `json:"tool_name,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type ollamaToolCall struct {
	ID       string                 `json:"id,omitempty"`
	Type     string                 `json:"type,omitempty"`
	Function ollamaToolFunctionCall `json:"function"`
}

type ollamaToolFunctionCall struct {
	Name      string `json:"name"`
	Arguments any    `json:"arguments,omitempty"`
}

type ollamaToolDef struct {
	Type     string                    `json:"type"`
	Function ollamaToolFunctionDefBody `json:"function"`
}

type ollamaToolFunctionDefBody struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type ollamaChatOptions struct {
	Temperature *float64 `json:"temperature,omitempty"`
	NumPredict  *int     `json:"num_predict,omitempty"`
}

func runChatFlow(
	ctx context.Context,
	c *client,
	modelName string,
	cfg model.GeneratorConfig,
	initialMessages []ollamasdk.ChatMessage,
	tools []model.Tool,
	handlers map[string]toolHandler,
) (string, flowUsageTotals, error) {
	history := make([]ollamaChatMessage, 0, len(initialMessages)+2)
	for _, message := range initialMessages {
		history = append(history, ollamaChatMessage{
			Role:    message.Role,
			Content: message.Content,
		})
	}

	toolDefs := buildOllamaToolDefs(tools)
	options := buildOllamaChatOptions(cfg)
	totals := flowUsageTotals{}

	for round := 0; round < maxToolRounds; round++ {
		response, err := c.chat(ctx, ollamaChatRequest{
			Model:    modelName,
			Messages: history,
			Stream:   false,
			Tools:    toolDefs,
			Options:  options,
		})
		if err != nil {
			return "", totals, utils.WrapIfNotNil(err)
		}

		totals.APICalls++
		totals.InputTokens += response.PromptEvalCount
		totals.OutputTokens += response.EvalCount
		totals.TotalTokens += response.PromptEvalCount + response.EvalCount

		assistantMessage := response.Message
		if strings.TrimSpace(assistantMessage.Role) == "" {
			assistantMessage.Role = "assistant"
		}
		assistantMessage.Content = strings.TrimSpace(assistantMessage.Content)

		toolCalls := assistantMessage.ToolCalls
		if len(tools) == 0 {
			return assistantMessage.Content, totals, nil
		}
		if len(toolCalls) == 0 {
			return assistantMessage.Content, totals, nil
		}

		history = append(history, assistantMessage)
		totals.ToolRounds = round + 1

		for _, toolCall := range toolCalls {
			handlerName, handler, err := resolveToolHandler(toolCall.Function.Name, handlers)
			if err != nil {
				return "", totals, utils.WrapIfNotNil(err)
			}

			argsBytes, err := normalizeToolArguments(toolCall.Function.Arguments)
			if err != nil {
				return "", totals, utils.WrapIfNotNil(err)
			}

			result, callErr := handler(ctx, argsBytes)
			resultPayload := any(result)
			if callErr != nil {
				resultPayload = map[string]any{
					"error": callErr.Error(),
				}
			}
			resultBytes, err := json.Marshal(resultPayload)
			if err != nil {
				return "", totals, utils.WrapIfNotNil(err)
			}

			history = append(history, ollamaChatMessage{
				Role:       "tool",
				Content:    string(resultBytes),
				Name:       handlerName,
				ToolName:   handlerName,
				ToolCallID: toolCall.ID,
			})
		}
	}

	return "", totals, utils.WrapIfNotNil(fmt.Errorf("exceeded tool call loop limit (%d)", maxToolRounds))
}

func (c *client) chat(ctx context.Context, request ollamaChatRequest) (*ollamaChatResponse, error) {
	body, err := json.Marshal(request)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	httpRequest, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		strings.TrimRight(c.baseURL, "/")+"/api/chat",
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Accept", "application/json")

	httpClient := &http.Client{Timeout: 180 * time.Second}
	httpResponse, err := httpClient.Do(httpRequest)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}
	defer httpResponse.Body.Close()

	rawBody, err := io.ReadAll(httpResponse.Body)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	if httpResponse.StatusCode < http.StatusOK || httpResponse.StatusCode >= http.StatusMultipleChoices {
		var apiError ollamaErrorResponse
		if unmarshalErr := json.Unmarshal(rawBody, &apiError); unmarshalErr == nil && strings.TrimSpace(apiError.Error) != "" {
			return nil, utils.WrapIfNotNil(
				fmt.Errorf("ollama chat request failed with status %d: %s", httpResponse.StatusCode, apiError.Error),
			)
		}
		return nil, utils.WrapIfNotNil(
			fmt.Errorf("ollama chat request failed with status %d: %s", httpResponse.StatusCode, strings.TrimSpace(string(rawBody))),
		)
	}

	var response ollamaChatResponse
	if err := json.Unmarshal(rawBody, &response); err != nil {
		return nil, utils.WrapIfNotNil(err)
	}
	if strings.TrimSpace(response.Error) != "" {
		return nil, utils.WrapIfNotNil(errors.New(strings.TrimSpace(response.Error)))
	}

	return &response, nil
}

func applyOllamaMetadata(meta model.GenerationMetadata, totals flowUsageTotals) {
	if meta == nil {
		return
	}

	meta[model.MetadataKeyAPICalls] = fmt.Sprintf("%d", totals.APICalls)
	meta[model.MetadataKeyToolRounds] = fmt.Sprintf("%d", totals.ToolRounds)
	meta[model.MetadataKeyInputTokens] = fmt.Sprintf("%d", totals.InputTokens)
	meta[model.MetadataKeyOutputTokens] = fmt.Sprintf("%d", totals.OutputTokens)
	meta[model.MetadataKeyTotalTokens] = fmt.Sprintf("%d", totals.TotalTokens)
}

func buildOllamaToolDefs(tools []model.Tool) []ollamaToolDef {
	if len(tools) == 0 {
		return nil
	}

	out := make([]ollamaToolDef, 0, len(tools))
	for _, tool := range tools {
		parameters := map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}
		if tool.InputSchema != nil {
			parameters = map[string]any(tool.InputSchema)
		}

		out = append(out, ollamaToolDef{
			Type: "function",
			Function: ollamaToolFunctionDefBody{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  parameters,
			},
		})
	}

	return out
}

func buildOllamaChatOptions(cfg model.GeneratorConfig) *ollamaChatOptions {
	if cfg.Temperature == nil && cfg.MaxTokens == nil {
		return nil
	}

	options := &ollamaChatOptions{}
	if cfg.Temperature != nil {
		temperature := *cfg.Temperature
		options.Temperature = &temperature
	}
	if cfg.MaxTokens != nil {
		numPredict := *cfg.MaxTokens
		options.NumPredict = &numPredict
	}
	return options
}

func resolveToolHandler(name string, handlers map[string]toolHandler) (string, toolHandler, error) {
	candidate := strings.TrimSpace(name)
	if candidate == "" {
		return "", nil, utils.WrapIfNotNil(errors.New("tool call name is required"))
	}

	if handler, ok := handlers[candidate]; ok {
		return candidate, handler, nil
	}

	for _, prefix := range []string{"tool.", "function.", "functions."} {
		trimmed := strings.TrimPrefix(candidate, prefix)
		if trimmed == candidate {
			continue
		}
		if handler, ok := handlers[trimmed]; ok {
			return trimmed, handler, nil
		}
	}

	return "", nil, utils.WrapIfNotNil(fmt.Errorf("no tool handler configured for function %q", candidate))
}

func normalizeToolArguments(arguments any) (json.RawMessage, error) {
	if arguments == nil {
		return json.RawMessage(`{}`), nil
	}

	switch value := arguments.(type) {
	case string:
		trimmed := strings.TrimSpace(value)
		if trimmed == "" {
			return json.RawMessage(`{}`), nil
		}
		if !json.Valid([]byte(trimmed)) {
			return nil, utils.WrapIfNotNil(fmt.Errorf("tool arguments are not valid JSON: %q", trimmed))
		}
		return json.RawMessage(trimmed), nil
	case json.RawMessage:
		trimmed := bytes.TrimSpace(value)
		if len(trimmed) == 0 {
			return json.RawMessage(`{}`), nil
		}
		if !json.Valid(trimmed) {
			return nil, utils.WrapIfNotNil(errors.New("tool arguments are not valid JSON"))
		}
		return json.RawMessage(trimmed), nil
	default:
		encoded, err := json.Marshal(arguments)
		if err != nil {
			return nil, utils.WrapIfNotNil(err)
		}
		encoded = bytes.TrimSpace(encoded)
		if len(encoded) == 0 {
			return json.RawMessage(`{}`), nil
		}
		return json.RawMessage(encoded), nil
	}
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

func (g *structuredGenerator[T]) repairStructuredJSON(
	ctx context.Context,
	modelName string,
	schema map[string]any,
	rawOutput string,
) (string, error) {
	schemaBytes, err := json.Marshal(schema)
	if err != nil {
		return "", utils.WrapIfNotNil(err)
	}

	messages := []ollamasdk.ChatMessage{
		{
			Role:    "system",
			Content: "You are a strict JSON formatter.",
		},
		{
			Role: "user",
			Content: "Reformat the following output into valid JSON matching this schema. Return only JSON.\n\n" +
				"Schema:\n" + string(schemaBytes) + "\n\n" +
				"Output:\n" + rawOutput,
		},
	}

	text, err := g.client.apiClient.Chat(modelName, messages)
	if err != nil {
		return "", utils.WrapIfNotNil(err)
	}
	return strings.TrimSpace(text), nil
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
