package gemini

import (
	"context"
	"errors"
	"strings"
	"sync"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/logging"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
)

type Client[T any] struct {
	cfg model.GeneratorConfig
}

func NewStructureContentGenerator[T any](prompt string, opts ...model.GeneratorOption) (model.ContentGenerator[T], error) {
	const fn = "gemini.NewStructureContentGenerator"
	cfg := model.ResolveGeneratorOpts(opts...)
	if prompt == "" {
		return nil, utils.WrapIfNotNil(errors.New("prompt is required"), fn)
	}
	return &structuredGenerator[T]{prompt: prompt, cfg: cfg}, nil
}

func NewStringContentGenerator(prompt string, opts ...model.GeneratorOption) (model.ContentGenerator[string], error) {
	const fn = "gemini.NewStringContentGenerator"
	cfg := model.ResolveGeneratorOpts(opts...)
	if prompt == "" {
		return nil, utils.WrapIfNotNil(errors.New("prompt is required"), fn)
	}
	return &textGenerator{prompt: prompt, cfg: cfg}, nil
}

type structuredGenerator[T any] struct {
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

func (g *structuredGenerator[T]) Generate(ctx context.Context) (T, error) {
	const fn = "gemini.structuredGenerator.Generate"
	log := logging.NewLogger(ctx)
	prompt, contextCount, err := g.promptWithContext(ctx)
	if err != nil {
		log.Errorf("%s error: %v", fn, err)
		var zero T
		return zero, utils.WrapIfNotNil(err, fn)
	}
	log.Infof(
		"%s prompt=%q context_count=%d temperature=%v max_tokens=%v tools=%d url=%q auth_token_set=%t",
		fn,
		prompt,
		contextCount,
		g.cfg.Temperature,
		g.cfg.MaxTokens,
		len(g.cfg.Tools),
		g.cfg.URL,
		g.cfg.AuthToken != "",
	)

	var zero T
	err = errors.New("gemini structured generation not implemented")
	log.Errorf("%s error: %v", fn, err)
	return zero, utils.WrapIfNotNil(err)
}

type textGenerator struct {
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

func (g *textGenerator) Generate(ctx context.Context) (string, error) {
	const fn = "gemini.textGenerator.Generate"
	log := logging.NewLogger(ctx)
	prompt, contextCount, err := g.promptWithContext(ctx)
	if err != nil {
		log.Errorf("%s error: %v", fn, err)
		return "", utils.WrapIfNotNil(err, fn)
	}
	log.Infof(
		"%s prompt=%q context_count=%d temperature=%v max_tokens=%v tools=%d url=%q auth_token_set=%t",
		fn,
		prompt,
		contextCount,
		g.cfg.Temperature,
		g.cfg.MaxTokens,
		len(g.cfg.Tools),
		g.cfg.URL,
		g.cfg.AuthToken != "",
	)

	err = errors.New("gemini text generation not implemented")
	log.Errorf("%s error: %v", fn, err)
	return "", utils.WrapIfNotNil(err)
}

func (g *structuredGenerator[T]) promptWithContext(ctx context.Context) (string, int, error) {
	const fn = "gemini.structuredGenerator.promptWithContext"
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
	const fn = "gemini.textGenerator.promptWithContext"
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
