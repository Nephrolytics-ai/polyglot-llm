package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeToolClient struct {
	initializeResult *mcp.InitializeResult
	initializeErr    error
	listToolsResult  *mcp.ListToolsResult
	listToolsErr     error
	callToolResult   *mcp.CallToolResult
	callToolErr      error
	closeErr         error

	lastCallRequest *mcp.CallToolRequest
}

func (f *fakeToolClient) Initialize(ctx context.Context, request mcp.InitializeRequest) (*mcp.InitializeResult, error) {
	return f.initializeResult, f.initializeErr
}

func (f *fakeToolClient) ListTools(ctx context.Context, request mcp.ListToolsRequest) (*mcp.ListToolsResult, error) {
	return f.listToolsResult, f.listToolsErr
}

func (f *fakeToolClient) CallTool(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	reqCopy := request
	f.lastCallRequest = &reqCopy
	return f.callToolResult, f.callToolErr
}

func (f *fakeToolClient) Close() error {
	return f.closeErr
}

func TestSchemaToMapUsesRawInputSchemaWhenPresent(t *testing.T) {
	tool := mcp.Tool{
		Name:           "raw_schema_tool",
		RawInputSchema: json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`),
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"ignored": map[string]any{"type": "string"},
			},
		},
	}

	schema, err := schemaToMap(tool)
	require.NoError(t, err)
	assert.Equal(t, "object", schema["type"])
	props, ok := schema["properties"].(map[string]any)
	require.True(t, ok)
	_, hasX := props["x"]
	assert.True(t, hasX)
	_, hasIgnored := props["ignored"]
	assert.False(t, hasIgnored)
}

func TestSchemaToMapFallsBackToInputSchema(t *testing.T) {
	tool := mcp.Tool{
		Name: "input_schema_tool",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"city": map[string]any{"type": "string"},
			},
			Required: []string{"city"},
		},
	}

	schema, err := schemaToMap(tool)
	require.NoError(t, err)
	assert.Equal(t, "object", schema["type"])
	props, ok := schema["properties"].(map[string]any)
	require.True(t, ok)
	_, hasCity := props["city"]
	assert.True(t, hasCity)
}

func TestFilterAllowedTools(t *testing.T) {
	adapter := &ToolAdapter{
		allowedTools: normalizeAllowedTools([]string{"a"}),
	}

	filtered := adapter.filterAllowedTools([]mcp.Tool{
		{Name: "a"},
		{Name: "b"},
	})

	require.Len(t, filtered, 1)
	assert.Equal(t, "a", filtered[0].Name)
}

func TestAsModelToolsAndExecuteTool(t *testing.T) {
	fake := &fakeToolClient{
		callToolResult: &mcp.CallToolResult{
			IsError:           false,
			StructuredContent: map[string]any{"result": "ok"},
			Content:           []mcp.Content{mcp.NewTextContent("done")},
		},
	}

	adapter := &ToolAdapter{
		serverURL:       "https://example.com/mcp",
		serverAuthToken: "Bearer token123",
		client:          fake,
		tools: []mcp.Tool{
			{
				Name:           "echo",
				Description:    "echoes a value",
				RawInputSchema: json.RawMessage(`{"type":"object","properties":{"value":{"type":"string"}}}`),
			},
		},
	}

	modelTools, err := adapter.AsModelTools()
	require.NoError(t, err)
	require.Len(t, modelTools, 1)

	out, err := modelTools[0].Handler(context.Background(), json.RawMessage(`{"value":"hello"}`))
	require.NoError(t, err)

	outMap, ok := out.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, false, outMap["is_error"])

	require.NotNil(t, fake.lastCallRequest)
	assert.Equal(t, "echo", fake.lastCallRequest.Params.Name)
	assert.Equal(t, "Bearer token123", fake.lastCallRequest.Header.Get("Authorization"))

	args, ok := fake.lastCallRequest.Params.Arguments.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "hello", args["value"])
}

func TestExecuteToolCallErrorIsReturnedAsPayload(t *testing.T) {
	fake := &fakeToolClient{
		callToolErr: errors.New("call failed"),
	}

	adapter := &ToolAdapter{
		serverURL: "https://example.com/mcp",
		client:    fake,
	}

	out, err := adapter.ExecuteTool(context.Background(), "echo", nil)
	require.NoError(t, err)
	outMap, ok := out.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, true, outMap["is_error"])
	assert.Contains(t, outMap["error"], "call failed")
}

func TestExecuteToolInvalidArgumentsReturnError(t *testing.T) {
	adapter := &ToolAdapter{
		serverURL: "https://example.com/mcp",
		client:    &fakeToolClient{},
	}

	_, err := adapter.ExecuteTool(context.Background(), "echo", json.RawMessage(`{"value":`))
	require.Error(t, err)
}
