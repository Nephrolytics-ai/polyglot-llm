package huggingface

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
	s.Equal("function", tools[0].Type)
	s.Equal("echo", tools[0].Function.Name)
	s.Equal("echo input", tools[0].Function.Description)
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

func (s *ToolsSuite) TestMapLocalToolsMissingHandler() {
	_, _, err := mapLocalTools([]model.Tool{
		{Name: "no-handler"},
	})

	s.Error(err)
	s.Contains(err.Error(), "tool handler is required")
}

func (s *ToolsSuite) TestMapLocalToolsMissingName() {
	_, _, err := mapLocalTools([]model.Tool{
		{Handler: func(ctx context.Context, args json.RawMessage) (any, error) { return nil, nil }},
	})

	s.Error(err)
	s.Contains(err.Error(), "tool name is required")
}

func (s *ToolsSuite) TestMapLocalToolsDefaultSchema() {
	tools, _, err := mapLocalTools([]model.Tool{
		{
			Name:    "simple",
			Handler: func(ctx context.Context, args json.RawMessage) (any, error) { return nil, nil },
		},
	})

	s.Require().NoError(err)
	s.Len(tools, 1)
	s.Equal("object", tools[0].Function.Parameters["type"])
}

func (s *ToolsSuite) TestExtractAuthorizationHeader() {
	s.Equal("Bearer tok", extractAuthorizationHeader(map[string]string{
		"authorization": "Bearer tok",
	}))
}

func (s *ToolsSuite) TestExtractAuthorizationHeaderMissing() {
	s.Equal("", extractAuthorizationHeader(map[string]string{
		"X-Custom": "val",
	}))
}

func (s *ToolsSuite) TestExtractAuthorizationHeaderEmpty() {
	s.Equal("", extractAuthorizationHeader(nil))
}
