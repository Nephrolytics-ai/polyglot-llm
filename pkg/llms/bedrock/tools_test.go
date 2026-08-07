package bedrock

import (
	"testing"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/stretchr/testify/assert"
)

func TestResolveMCPAuthorizationUsesAuthTokenWhenHeaderMissing(t *testing.T) {
	value := resolveMCPAuthorization(model.MCPTool{
		AuthToken: "Bearer auth-token",
	})

	assert.Equal(t, "Bearer auth-token", value)
}

func TestResolveMCPAuthorizationUsesAuthTokenWhenAuthorizationHeaderAbsent(t *testing.T) {
	value := resolveMCPAuthorization(model.MCPTool{
		AuthToken: "Bearer auth-token",
		HTTPHeaders: map[string]string{
			"X-Custom": "value",
		},
	})

	assert.Equal(t, "Bearer auth-token", value)
}

func TestResolveMCPAuthorizationPrefersExplicitAuthorizationHeader(t *testing.T) {
	value := resolveMCPAuthorization(model.MCPTool{
		AuthToken: "Bearer auth-token",
		HTTPHeaders: map[string]string{
			"Authorization": "Bearer header-token",
		},
	})

	assert.Equal(t, "Bearer header-token", value)
}

func TestResolveMCPAuthorizationFindsHeaderCaseInsensitively(t *testing.T) {
	value := resolveMCPAuthorization(model.MCPTool{
		AuthToken: "Bearer auth-token",
		HTTPHeaders: map[string]string{
			"authorization": "Bearer lower-header-token",
		},
	})

	assert.Equal(t, "Bearer lower-header-token", value)
}

func TestResolveMCPAuthorizationFallsBackWhenAuthorizationHeaderBlank(t *testing.T) {
	value := resolveMCPAuthorization(model.MCPTool{
		AuthToken: "Bearer auth-token",
		HTTPHeaders: map[string]string{
			"Authorization": "   ",
		},
	})

	assert.Equal(t, "Bearer auth-token", value)
}
