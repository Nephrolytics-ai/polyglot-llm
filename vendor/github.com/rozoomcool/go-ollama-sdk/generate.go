package ollama

import (
	"encoding/json"
	"io"
)

// Generate делает синхронную генерацию
func (c *OllamaClient) Generate(model, prompt string) (string, error) {
	req := GenerateRequest{
		Model:  model,
		Prompt: prompt,
		Stream: false,
	}

	resp, err := c.post("/api/generate", req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result GenerateResponse
	body, _ := io.ReadAll(resp.Body)
	err = json.Unmarshal(body, &result)
	if err != nil {
		return "", err
	}
	return result.Response, nil
}

// GenerateStream — стриминговая генерация с колбэком
func (c *OllamaClient) GenerateStream(model, prompt string, onData func(string), onDone func()) error {
	req := GenerateRequest{
		Model:  model,
		Prompt: prompt,
		Stream: true,
	}

	resp, err := c.post("/api/generate", req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	decoder := json.NewDecoder(resp.Body)
	for decoder.More() {
		var chunk GenerateResponse
		if err := decoder.Decode(&chunk); err != nil {
			return err
		}
		onData(chunk.Response)
		if chunk.Done {
			onDone()
			break
		}
	}
	return nil
}
