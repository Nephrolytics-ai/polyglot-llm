package mcp

import (
	"context"
	"fmt"
	"sync"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
)

var cachedTools []string = nil
var cachedToolsMutex sync.RWMutex

func FetchListOfTools(ctx context.Context, serverURL string, authToken string) ([]string, error) {

	cachedToolsMutex.RLock()
	tmpTools := cachedTools
	cachedToolsMutex.RUnlock()
	if tmpTools != nil {
		return tmpTools, nil
	}
	cachedToolsMutex.Lock()
	defer cachedToolsMutex.Unlock()
	if cachedTools != nil {
		return cachedTools, nil
	}
	tmpTools, err := actuallyFetchListOfTools(ctx, serverURL, authToken)
	if err != nil {
		return nil, err
	}
	cachedTools = tmpTools
	return tmpTools, nil
}
func actuallyFetchListOfTools(ctx context.Context, serverURL string, authToken string) ([]string, error) {

	headers := make(map[string]string)
	headers["Authorization"] = authToken
	httpTransport, err := transport.NewStreamableHTTP(serverURL, transport.WithHTTPHeaders(headers))

	// Create client with the transport
	c := client.NewClient(httpTransport)
	defer c.Close()

	// Initialize the client
	fmt.Println("Initializing client...")
	initRequest := mcp.InitializeRequest{}
	initRequest.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION

	initRequest.Params.ClientInfo = mcp.Implementation{
		Name:    "Nephrolytics AI Helper",
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
