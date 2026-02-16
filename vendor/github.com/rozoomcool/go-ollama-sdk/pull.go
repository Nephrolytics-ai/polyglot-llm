package ollama

import (
	"encoding/json"
)

// PullModel — скачивание модели
func (c *OllamaClient) PullModel(name string, onStatus func(string)) error {
	req := PullRequest{Name: name}

	resp, err := c.post("/api/pull", req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	decoder := json.NewDecoder(resp.Body)
	for decoder.More() {
		var chunk PullResponse
		if err := decoder.Decode(&chunk); err != nil {
			return err
		}
		onStatus(chunk.Status)
	}
	return nil
}
