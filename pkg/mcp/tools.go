package mcp

import (
	"context"
	"sync"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
)

var cachedToolsByURL = map[string][]string{}
var cachedToolsMutex sync.RWMutex

func FetchListOfTools(ctx context.Context, serverURL string, authToken string) ([]string, error) {
	cachedToolsMutex.RLock()
	tmpTools, found := cachedToolsByURL[serverURL]
	cachedToolsMutex.RUnlock()
	if found {
		return append([]string(nil), tmpTools...), nil
	}

	cachedToolsMutex.Lock()
	defer cachedToolsMutex.Unlock()

	tmpTools, found = cachedToolsByURL[serverURL]
	if found {
		return append([]string(nil), tmpTools...), nil
	}

	tmpTools, err := actuallyFetchListOfTools(ctx, serverURL, authToken)
	if err != nil {
		return nil, err
	}

	cachedToolsByURL[serverURL] = append([]string(nil), tmpTools...)
	return append([]string(nil), tmpTools...), nil
}
func actuallyFetchListOfTools(ctx context.Context, serverURL string, authToken string) ([]string, error) {

	headers := make(map[string]string)
	headers["Authorization"] = authToken
	httpTransport, err := transport.NewStreamableHTTP(serverURL, transport.WithHTTPHeaders(headers))

	// Create client with the transport
	c := client.NewClient(httpTransport)
	defer c.Close()

	// Initialize the client
	initRequest := mcp.InitializeRequest{}
	initRequest.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION

	initRequest.Params.ClientInfo = mcp.Implementation{
		Name:    "Polyglot LLM AI Helper",
		Version: "1.0.0",
	}
	initRequest.Params.Capabilities = mcp.ClientCapabilities{}

	serverInfo, err := c.Initialize(ctx, initRequest)
	if err != nil {
		return nil, utils.WrapIfNotNil(err, "Fetching List of tools Init Failed")
	}
	ret := make([]string, 0)
	// List available tools if the server supports them
	if serverInfo.Capabilities.Tools != nil {
		toolsRequest := mcp.ListToolsRequest{}
		toolsResult, err := c.ListTools(ctx, toolsRequest)
		if err != nil {
			return nil, utils.WrapIfNotNil(err, "Fetching List of tools list Failed")
		}
		for _, tool := range toolsResult.Tools {
			ret = append(ret, tool.Name)
		}
	}

	return ret, nil
}
