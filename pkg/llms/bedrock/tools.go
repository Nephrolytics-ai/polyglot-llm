package bedrock

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/logging"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/mcp"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
	"github.com/aws/aws-sdk-go-v2/aws"
	bedrockdocument "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

type toolHandler func(ctx context.Context, args []byte) (any, error)

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

func mapTools(tools []model.Tool) (*bedrocktypes.ToolConfiguration, map[string]toolHandler, error) {
	if len(tools) == 0 {
		return nil, nil, nil
	}

	mappedTools := make([]bedrocktypes.Tool, 0, len(tools))
	handlers := make(map[string]toolHandler, len(tools))

	for _, tool := range tools {
		if strings.TrimSpace(tool.Name) == "" {
			return nil, nil, utils.WrapIfNotNil(errors.New("tool name is required"))
		}
		if tool.Handler == nil {
			return nil, nil, utils.WrapIfNotNil(fmt.Errorf("tool handler is required for %q", tool.Name))
		}
		if _, exists := handlers[tool.Name]; exists {
			return nil, nil, utils.WrapIfNotNil(fmt.Errorf("duplicate tool name %q", tool.Name))
		}

		parameters := map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}
		if tool.InputSchema != nil {
			parameters = map[string]any(tool.InputSchema)
		}

		mappedTools = append(mappedTools, &bedrocktypes.ToolMemberToolSpec{
			Value: bedrocktypes.ToolSpecification{
				Name: aws.String(tool.Name),
				InputSchema: &bedrocktypes.ToolInputSchemaMemberJson{
					Value: bedrockdocument.NewLazyDocument(parameters),
				},
			},
		})

		toolHandlerFunc := tool.Handler
		handlers[tool.Name] = func(ctx context.Context, args []byte) (any, error) {
			return toolHandlerFunc(ctx, args)
		}
	}

	return &bedrocktypes.ToolConfiguration{
		Tools: mappedTools,
	}, handlers, nil
}

func extractAuthorizationHeader(headers map[string]string) string {
	for k, v := range headers {
		if strings.EqualFold(k, "Authorization") {
			return v
		}
	}
	return ""
}
