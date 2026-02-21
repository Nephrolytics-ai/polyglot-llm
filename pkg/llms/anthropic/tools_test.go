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

func (s *ToolsSuite) TestMapMCPServersAuthTokenAndAllowedTools() {
	servers, err := mapMCPServers(context.Background(), []model.MCPTool{
		{
			Name:      "mcp-a",
			URL:       "https://example-mcp",
			AuthToken: "Bearer abc123",
			HTTPHeaders: map[string]string{
				"X-Custom": "ignored",
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
}

func (s *ToolsSuite) TestMapMCPServersIgnoresAuthorizationHeaderWithoutAuthToken() {
	servers, err := mapMCPServers(context.Background(), []model.MCPTool{
		{
			Name: "mcp-a",
			URL:  "https://example-mcp",
			HTTPHeaders: map[string]string{
				"Authorization": "Bearer from-header",
			},
		},
	})

	s.Require().NoError(err)
	s.Len(servers, 1)
	s.Empty(servers[0].AuthorizationToken)
}

func (s *ToolsSuite) TestMapMCPToolsetsCreatesServerReferences() {
	toolsets, err := mapMCPToolsets([]model.MCPTool{
		{
			Name: "dev_lab_mcp",
			URL:  "https://example-mcp",
		},
	})

	s.Require().NoError(err)
	s.Len(toolsets, 1)
	s.Equal("mcp_toolset", toolsets[0].Type)
	s.Empty(toolsets[0].Name)
	s.Equal("dev_lab_mcp", toolsets[0].MCPServerName)
}

func (s *ToolsSuite) TestMapMCPToolsetsAllowedToolsConfig() {
	toolsets, err := mapMCPToolsets([]model.MCPTool{
		{
			Name:         "dev_lab_mcp",
			URL:          "https://example-mcp",
			AllowedTools: []string{"lab_read", "lab_read", " patient_search "},
		},
	})

	s.Require().NoError(err)
	s.Len(toolsets, 1)
	s.NotNil(toolsets[0].DefaultConfig)
	s.Require().NotNil(toolsets[0].DefaultConfig.Enabled)
	s.False(*toolsets[0].DefaultConfig.Enabled)
	s.Require().Len(toolsets[0].Configs, 2)
	labReadCfg, ok := toolsets[0].Configs["lab_read"]
	s.True(ok)
	s.Require().NotNil(labReadCfg.Enabled)
	s.True(*labReadCfg.Enabled)
	patientSearchCfg, ok := toolsets[0].Configs["patient_search"]
	s.True(ok)
	s.Require().NotNil(patientSearchCfg.Enabled)
	s.True(*patientSearchCfg.Enabled)
}
