package ollama

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

type OllamaClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient создает новый клиент Ollama
func NewClient(baseURL string) *OllamaClient {
	return &OllamaClient{
		baseURL:    baseURL,
		httpClient: &http.Client{},
	}
}

// internal helper
func (c *OllamaClient) post(path string, request interface{}) (*http.Response, error) {
	data, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}
	url := fmt.Sprintf("%s%s", c.baseURL, path)
	resp, err := c.httpClient.Post(url, "application/json", bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("%w: %s", ErrRequestFailed, resp.Status)
	}
	return resp, nil
}
