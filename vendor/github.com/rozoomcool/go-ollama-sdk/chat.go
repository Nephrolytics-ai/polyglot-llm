package ollama

import (
	"encoding/json"
	"io"
)

// Chat делает синхронный чат-запрос
func (c *OllamaClient) Chat(model string, messages []ChatMessage) (string, error) {
	req := ChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   false,
	}

	resp, err := c.post("/api/chat", req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result ChatResponse
	body, _ := io.ReadAll(resp.Body)
	err = json.Unmarshal(body, &result)
	if err != nil {
		return "", err
	}
	return result.Message.Content, nil
}

// ChatStream — стриминговый чат с колбэком
func (c *OllamaClient) ChatStream(model string, messages []ChatMessage, onData func(string), onDone func()) error {
	req := ChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   true,
	}

	resp, err := c.post("/api/chat", req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	decoder := json.NewDecoder(resp.Body)
	for decoder.More() {
		var chunk ChatResponse
		if err := decoder.Decode(&chunk); err != nil {
			return err
		}
		onData(chunk.Message.Content)
		if chunk.Done {
			onDone()
			break
		}
	}
	return nil
}
