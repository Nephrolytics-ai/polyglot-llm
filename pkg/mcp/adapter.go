package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
)

type toolClient interface {
	Initialize(ctx context.Context, request mcp.InitializeRequest) (*mcp.InitializeResult, error)
	ListTools(ctx context.Context, request mcp.ListToolsRequest) (*mcp.ListToolsResult, error)
	CallTool(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error)
	Close() error
}

// ToolAdapter bridges MCP tools into local model.Tool definitions for providers
// that do not support MCP natively.
type ToolAdapter struct {
	serverURL       string
	serverAuthToken string
	allowedTools    map[string]struct{}

	mu     sync.RWMutex
	client toolClient
	tools  []mcp.Tool
}

func NewToolAdapter(ctx context.Context, serverURL string, authToken string, allowedTools []string) (*ToolAdapter, error) {
	a := &ToolAdapter{
		serverURL:       serverURL,
		serverAuthToken: authToken,
		allowedTools:    normalizeAllowedTools(allowedTools),
	}
	err := a.Connect(ctx)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}
	return a, nil
}

func (a *ToolAdapter) Connect(ctx context.Context) error {
	if strings.TrimSpace(a.serverURL) == "" {
		return utils.WrapIfNotNil(errors.New("serverURL is required"))
	}

	headers := map[string]string{}
	if a.serverAuthToken != "" {
		headers["Authorization"] = a.serverAuthToken
	}

	httpTransport, err := transport.NewStreamableHTTP(
		a.serverURL,
		transport.WithHTTPHeaders(headers),
	)
	if err != nil {
		return utils.WrapIfNotNil(err)
	}

	c := client.NewClient(httpTransport)
	tools, initErr := initializeAndListTools(ctx, c)
	if initErr != nil {
		_ = c.Close()
		return utils.WrapIfNotNil(initErr)
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	if a.client != nil {
		_ = a.client.Close()
	}

	a.client = c
	a.tools = a.filterAllowedTools(tools)
	return nil
}

func (a *ToolAdapter) RefreshTools(ctx context.Context) error {
	a.mu.RLock()
	c := a.client
	a.mu.RUnlock()

	if c == nil {
		return utils.WrapIfNotNil(errors.New("mcp client is not connected"))
	}

	toolsResult, err := c.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return utils.WrapIfNotNil(err)
	}

	tools := []mcp.Tool{}
	if toolsResult != nil {
		tools = toolsResult.Tools
	}

	a.mu.Lock()
	a.tools = a.filterAllowedTools(tools)
	a.mu.Unlock()
	return nil
}

func (a *ToolAdapter) Disconnect() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	var err error
	if a.client != nil {
		err = a.client.Close()
	}
	a.client = nil
	a.tools = nil
	if err != nil {
		return utils.WrapIfNotNil(err)
	}
	return nil
}

func (a *ToolAdapter) Tools() []mcp.Tool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return append([]mcp.Tool(nil), a.tools...)
}

func (a *ToolAdapter) AsModelTools() ([]model.Tool, error) {
	a.mu.RLock()
	tools := append([]mcp.Tool(nil), a.tools...)
	a.mu.RUnlock()

	out := make([]model.Tool, 0, len(tools))
	for _, mcpTool := range tools {
		schema, err := schemaToMap(mcpTool)
		if err != nil {
			return nil, utils.WrapIfNotNil(fmt.Errorf("tool %q schema conversion failed: %w", mcpTool.Name, err))
		}

		toolName := mcpTool.Name
		out = append(out, model.Tool{
			Name:        toolName,
			Description: mcpTool.Description,
			InputSchema: model.JSONSchema(schema),
			Handler: func(ctx context.Context, args json.RawMessage) (any, error) {
				return a.ExecuteTool(ctx, toolName, args)
			},
		})
	}
	return out, nil
}

func (a *ToolAdapter) ExecuteTool(ctx context.Context, toolName string, rawArgs json.RawMessage) (any, error) {
	a.mu.RLock()
	c := a.client
	authToken := a.serverAuthToken
	a.mu.RUnlock()

	if c == nil {
		return nil, utils.WrapIfNotNil(errors.New("mcp client is not connected"))
	}
	if strings.TrimSpace(toolName) == "" {
		return nil, utils.WrapIfNotNil(errors.New("toolName is required"))
	}

	args := map[string]any{}
	if len(rawArgs) > 0 && string(rawArgs) != "null" {
		err := json.Unmarshal(rawArgs, &args)
		if err != nil {
			return nil, utils.WrapIfNotNil(err)
		}
	}

	request := mcp.CallToolRequest{
		Header: http.Header{},
		Params: mcp.CallToolParams{
			Name:      toolName,
			Arguments: args,
		},
	}
	if authToken != "" {
		request.Header.Set("Authorization", authToken)
	}

	result, err := c.CallTool(ctx, request)
	if err != nil {
		// Preserve the failure as tool output so the model can see and recover.
		return map[string]any{
			"is_error": true,
			"error":    err.Error(),
		}, nil
	}

	normalized, normErr := normalizeCallToolResult(result)
	if normErr != nil {
		return map[string]any{
			"is_error": true,
			"error":    normErr.Error(),
		}, nil
	}
	return normalized, nil
}

func initializeAndListTools(ctx context.Context, c toolClient) ([]mcp.Tool, error) {
	initRequest := mcp.InitializeRequest{}
	initRequest.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initRequest.Params.ClientInfo = mcp.Implementation{
		Name:    "Nephrolytics MCP Tool Adapter",
		Version: "1.0.0",
	}
	initRequest.Params.Capabilities = mcp.ClientCapabilities{}

	serverInfo, err := c.Initialize(ctx, initRequest)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}
	if serverInfo == nil || serverInfo.Capabilities.Tools == nil {
		return nil, nil
	}

	toolsResult, err := c.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}
	if toolsResult == nil {
		return nil, nil
	}
	return toolsResult.Tools, nil
}

// schemaToMap returns a JSON-schema map for an MCP tool.
// Priority: RawInputSchema (if present) > InputSchema.
func schemaToMap(tool mcp.Tool) (map[string]any, error) {
	if len(tool.RawInputSchema) > 0 {
		var schema map[string]any
		err := json.Unmarshal(tool.RawInputSchema, &schema)
		if err != nil {
			return nil, utils.WrapIfNotNil(fmt.Errorf("invalid raw input schema: %w", err))
		}
		return schema, nil
	}

	schemaBytes, err := json.Marshal(tool.InputSchema)
	if err != nil {
		return nil, utils.WrapIfNotNil(fmt.Errorf("marshal input schema failed: %w", err))
	}

	var schema map[string]any
	err = json.Unmarshal(schemaBytes, &schema)
	if err != nil {
		return nil, utils.WrapIfNotNil(fmt.Errorf("unmarshal input schema failed: %w", err))
	}
	return schema, nil
}

func normalizeCallToolResult(result *mcp.CallToolResult) (map[string]any, error) {
	if result == nil {
		return nil, utils.WrapIfNotNil(errors.New("nil call tool result"))
	}

	return map[string]any{
		"is_error":           result.IsError,
		"content":            result.Content,
		"structured_content": result.StructuredContent,
	}, nil
}

func normalizeAllowedTools(names []string) map[string]struct{} {
	if len(names) == 0 {
		return nil
	}

	out := make(map[string]struct{}, len(names))
	for _, name := range names {
		trimmed := strings.TrimSpace(name)
		if trimmed == "" {
			continue
		}
		out[trimmed] = struct{}{}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func (a *ToolAdapter) filterAllowedTools(tools []mcp.Tool) []mcp.Tool {
	if len(a.allowedTools) == 0 {
		return append([]mcp.Tool(nil), tools...)
	}

	filtered := make([]mcp.Tool, 0, len(tools))
	for _, tool := range tools {
		if _, ok := a.allowedTools[tool.Name]; ok {
			filtered = append(filtered, tool)
		}
	}
	return filtered
}
