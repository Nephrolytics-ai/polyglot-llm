package anthropic

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/stretchr/testify/suite"
)

type ToolsSuite struct {
	suite.Suite
}

func TestToolsSuite(t *testing.T) {
	suite.Run(t, new(ToolsSuite))
}

func (s *ToolsSuite) TestMapLocalToolsSuccess() {
	tools, handlers, err := mapLocalTools([]model.Tool{
		{
			Name:        "echo",
			Description: "echo input",
			InputSchema: model.JSONSchema{"type": "object"},
			Handler: func(ctx context.Context, args json.RawMessage) (any, error) {
				return map[string]any{"ok": true}, nil
			},
		},
	})

	s.Require().NoError(err)
	s.Len(tools, 1)
	s.Equal("echo", tools[0].Name)
	s.NotNil(handlers["echo"])
}

func (s *ToolsSuite) TestMapLocalToolsDuplicateName() {
	_, _, err := mapLocalTools([]model.Tool{
		{Name: "dup", Handler: func(ctx context.Context, args json.RawMessage) (any, error) { return nil, nil }},
		{Name: "dup", Handler: func(ctx context.Context, args json.RawMessage) (any, error) { return nil, nil }},
	})

	s.Error(err)
	s.Contains(err.Error(), "duplicate tool name")
}

func (s *ToolsSuite) TestMapMCPServersAuthorizationAndAllowedTools() {
	servers, err := mapMCPServers(context.Background(), []model.MCPTool{
		{
			Name: "mcp-a",
			URL:  "https://example-mcp",
			HTTPHeaders: map[string]string{
				"Authorization": "Bearer abc123",
				"X-Custom":      "ignored",
			},
			AllowedTools: []string{"tool-a", "tool-a", " tool-b "},
		},
	})

	s.Require().NoError(err)
	s.Len(servers, 1)
	s.Equal("url", servers[0].Type)
	s.Equal("mcp-a", servers[0].Name)
	s.Equal("https://example-mcp", servers[0].URL)
	s.Equal("Bearer abc123", servers[0].AuthorizationToken)
	s.NotNil(servers[0].ToolConfiguration)
	s.ElementsMatch([]string{"tool-a", "tool-b"}, servers[0].ToolConfiguration.AllowedTools)
}
