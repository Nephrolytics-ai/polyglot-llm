package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/logging"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
)

type toolHandler func(ctx context.Context, args json.RawMessage) (any, error)

func buildAllTools(
	ctx context.Context,
	cfg model.GeneratorConfig,
) ([]anthropicTool, map[string]toolHandler, []anthropicMCPServer, func(), error) {
	localTools, handlers, err := mapLocalTools(cfg.Tools)
	if err != nil {
		return nil, nil, nil, func() {}, utils.WrapIfNotNil(err)
	}

	mcpServers, err := mapMCPServers(ctx, cfg.MCPTools)
	if err != nil {
		return nil, nil, nil, func() {}, utils.WrapIfNotNil(err)
	}

	mcpToolsets, err := mapMCPToolsets(cfg.MCPTools)
	if err != nil {
		return nil, nil, nil, func() {}, utils.WrapIfNotNil(err)
	}

	tools := make([]anthropicTool, 0, len(localTools)+len(mcpToolsets))
	tools = append(tools, localTools...)
	tools = append(tools, mcpToolsets...)

	return tools, handlers, mcpServers, func() {}, nil
}

func mapLocalTools(tools []model.Tool) ([]anthropicTool, map[string]toolHandler, error) {
	mapped := make([]anthropicTool, 0, len(tools))
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

		inputSchema := map[string]any{
			"type":                 "object",
			"properties":           map[string]any{},
			"additionalProperties": false,
		}
		if tool.InputSchema != nil {
			inputSchema = map[string]any(tool.InputSchema)
		}

		mappedTool := anthropicTool{
			Name:        name,
			Description: strings.TrimSpace(tool.Description),
			InputSchema: inputSchema,
		}
		mapped = append(mapped, mappedTool)
		handlers[name] = tool.Handler
	}

	return mapped, handlers, nil
}

func mapMCPServers(ctx context.Context, mcpTools []model.MCPTool) ([]anthropicMCPServer, error) {
	log := logging.NewLogger(ctx)
	servers := make([]anthropicMCPServer, 0, len(mcpTools))

	for _, mcpTool := range mcpTools {
		name := strings.TrimSpace(mcpTool.Name)
		if name == "" {
			return nil, utils.WrapIfNotNil(errors.New("mcp tool name is required"))
		}

		url := strings.TrimSpace(mcpTool.URL)
		if url == "" {
			return nil, utils.WrapIfNotNil(fmt.Errorf("mcp tool URL is required for %q", name))
		}

		authorizationToken := strings.TrimSpace(mcpTool.AuthToken)
		warnOnUnsupportedMCPHeaders(log, mcpTool.Name, mcpTool.HTTPHeaders)

		server := anthropicMCPServer{
			Type: "url",
			Name: name,
			URL:  url,
		}
		if authorizationToken != "" {
			server.AuthorizationToken = authorizationToken
		}

		servers = append(servers, server)
	}

	return servers, nil
}

func mapMCPToolsets(mcpTools []model.MCPTool) ([]anthropicTool, error) {
	toolsets := make([]anthropicTool, 0, len(mcpTools))
	for _, mcpTool := range mcpTools {
		name := strings.TrimSpace(mcpTool.Name)
		if name == "" {
			return nil, utils.WrapIfNotNil(errors.New("mcp tool name is required"))
		}
		url := strings.TrimSpace(mcpTool.URL)
		if url == "" {
			return nil, utils.WrapIfNotNil(fmt.Errorf("mcp tool URL is required for %q", name))
		}

		toolsets = append(toolsets, anthropicTool{
			Type:          "mcp_toolset",
			MCPServerName: name,
		})
		if allowedTools := normalizeAllowedTools(mcpTool.AllowedTools); len(allowedTools) > 0 {
			disabled := false
			configs := make(map[string]anthropicMCPToolConfig, len(allowedTools))
			for _, allowedTool := range allowedTools {
				enabled := true
				configs[allowedTool] = anthropicMCPToolConfig{Enabled: &enabled}
			}
			toolsets[len(toolsets)-1].DefaultConfig = &anthropicMCPToolConfig{Enabled: &disabled}
			toolsets[len(toolsets)-1].Configs = configs
		}
	}
	return toolsets, nil
}

func warnOnUnsupportedMCPHeaders(log logging.Logger, toolName string, headers map[string]string) {
	if log == nil || len(headers) == 0 {
		return
	}

	unsupported := make([]string, 0)
	for key := range headers {
		if strings.EqualFold(strings.TrimSpace(key), "Authorization") {
			continue
		}
		unsupported = append(unsupported, key)
	}
	if len(unsupported) == 0 {
		return
	}

	// NOTE: Anthropic MCP uses MCPTool.AuthToken for authorization_token forwarding.
	// Arbitrary custom headers are not supported for remote MCP servers.
	log.Warnf(
		"mcp tool %q has unsupported custom headers (%s); anthropic MCP uses MCPTool.AuthToken instead of HTTPHeaders",
		toolName,
		strings.Join(unsupported, ","),
	)
}

func normalizeAllowedTools(names []string) []string {
	if len(names) == 0 {
		return nil
	}

	seen := make(map[string]struct{}, len(names))
	out := make([]string, 0, len(names))
	for _, name := range names {
		trimmed := strings.TrimSpace(name)
		if trimmed == "" {
			continue
		}
		if _, exists := seen[trimmed]; exists {
			continue
		}
		seen[trimmed] = struct{}{}
		out = append(out, trimmed)
	}

	if len(out) == 0 {
		return nil
	}
	return out
}
