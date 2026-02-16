package ollama

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

func buildAllTools(ctx context.Context, cfg model.GeneratorConfig) ([]model.Tool, func(), error) {
	combined := append([]model.Tool(nil), cfg.Tools...)
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
			return nil, func() {}, utils.WrapIfNotNil(err)
		}
		adapters = append(adapters, adapter)

		adapterTools, err := adapter.AsModelTools()
		if err != nil {
			cleanup()
			return nil, func() {}, utils.WrapIfNotNil(err)
		}
		combined = append(combined, adapterTools...)
	}

	return combined, cleanup, nil
}

func mapTools(tools []model.Tool) ([]model.Tool, map[string]toolHandler, error) {
	if len(tools) == 0 {
		return nil, nil, nil
	}

	handlers := make(map[string]toolHandler, len(tools))
	out := make([]model.Tool, 0, len(tools))

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

		handler := tool.Handler
		handlers[name] = func(ctx context.Context, args json.RawMessage) (any, error) {
			return handler(ctx, args)
		}
		tool.Name = name
		out = append(out, tool)
	}

	return out, handlers, nil
}

func extractAuthorizationHeader(headers map[string]string) string {
	for k, v := range headers {
		if strings.EqualFold(k, "Authorization") {
			return v
		}
	}
	return ""
}
