package gemini

import (
	"context"
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

func extractAuthorizationHeader(headers map[string]string) string {
	for k, v := range headers {
		if strings.EqualFold(k, "Authorization") {
			return v
		}
	}
	return ""
}
