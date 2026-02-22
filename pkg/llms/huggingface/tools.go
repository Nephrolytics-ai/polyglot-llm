package huggingface

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/logging"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/mcp"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
)

type toolHandler func(ctx context.Context, args json.RawMessage) (any, error)

func buildAllTools(ctx context.Context, cfg model.GeneratorConfig) ([]chatTool, map[string]toolHandler, func(), error) {
	localTools, handlers, err := mapLocalTools(cfg.Tools)
	if err != nil {
		return nil, nil, func() {}, utils.WrapIfNotNil(err)
	}

	adapters := make([]*mcp.ToolAdapter, 0, len(cfg.MCPTools))

	cleanup := func() {
		log := logging.NewLogger(ctx)
		for _, adapter := range adapters {
			if adapter == nil {
				continue
			}
			if err := adapter.Disconnect(); err != nil {
				log.Warnf("mcp adapter disconnect failed: %v", err)
			}
		}
	}

	for _, mcpTool := range cfg.MCPTools {
		authToken := extractAuthorizationHeader(mcpTool.HTTPHeaders)

		adapter, err := mcp.NewToolAdapter(ctx, mcpTool.URL, authToken, mcpTool.AllowedTools)
		if err != nil {
			cleanup()
			return nil, nil, func() {}, utils.WrapIfNotNil(err)
		}
		adapters = append(adapters, adapter)

		adapterTools, err := adapter.AsModelTools()
		if err != nil {
			cleanup()
			return nil, nil, func() {}, utils.WrapIfNotNil(err)
		}

		for _, modelTool := range adapterTools {
			ct, handler := convertModelToolToChatTool(modelTool)
			localTools = append(localTools, ct)
			handlers[modelTool.Name] = handler
		}
	}

	return localTools, handlers, cleanup, nil
}

func mapLocalTools(tools []model.Tool) ([]chatTool, map[string]toolHandler, error) {
	mapped := make([]chatTool, 0, len(tools))
	handlers := make(map[string]toolHandler, len(tools))

	for _, tool := range tools {
		name := strings.TrimSpace(tool.Name)
		if name == "" {
			return nil, nil, utils.WrapIfNotNil(errors.New("tool name is required"))
		}
		if tool.Handler == nil {
			return nil, nil, utils.WrapIfNotNil(fmt.Errorf("tool handler is required for %q", name))
		}
		if _, exists := handlers[name]; exists {
			return nil, nil, utils.WrapIfNotNil(fmt.Errorf("duplicate tool name %q", name))
		}

		parameters := map[string]any{
			"type":                 "object",
			"properties":           map[string]any{},
			"additionalProperties": false,
		}
		if tool.InputSchema != nil {
			parameters = map[string]any(tool.InputSchema)
		}

		mapped = append(mapped, chatTool{
			Type: "function",
			Function: chatFunction{
				Name:        name,
				Description: strings.TrimSpace(tool.Description),
				Parameters:  parameters,
			},
		})
		handlers[name] = tool.Handler
	}

	return mapped, handlers, nil
}

func convertModelToolToChatTool(tool model.Tool) (chatTool, toolHandler) {
	parameters := map[string]any{
		"type":                 "object",
		"properties":           map[string]any{},
		"additionalProperties": false,
	}
	if tool.InputSchema != nil {
		parameters = map[string]any(tool.InputSchema)
	}

	ct := chatTool{
		Type: "function",
		Function: chatFunction{
			Name:        strings.TrimSpace(tool.Name),
			Description: strings.TrimSpace(tool.Description),
			Parameters:  parameters,
		},
	}

	return ct, tool.Handler
}

func extractAuthorizationHeader(headers map[string]string) string {
	for k, v := range headers {
		if strings.EqualFold(k, "Authorization") {
			return v
		}
	}
	return ""
}
