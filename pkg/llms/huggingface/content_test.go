package huggingface

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/stretchr/testify/suite"
)

type ContentSuite struct {
	suite.Suite
}

func TestContentSuite(t *testing.T) {
	suite.Run(t, new(ContentSuite))
}

func (s *ContentSuite) TestBuildMessagesWithContext() {
	messages, contextCount, err := buildMessagesWithContext("final prompt", []*model.PromptContext{
		{
			MessageType: model.ContextMessageTypeSystem,
			Content:     "system one",
		},
		{
			MessageType: model.ContextMessageTypeHuman,
			Content:     "human context",
		},
		{
			MessageType: model.ContextMessageTypeAssistant,
			Content:     "assistant context",
		},
	})

	s.Require().NoError(err)
	s.Equal(3, contextCount)
	s.Len(messages, 4)
	s.Equal("system", messages[0].Role)
	s.Equal("system one", messages[0].Content)
	s.Equal("user", messages[1].Role)
	s.Equal("human context", messages[1].Content)
	s.Equal("assistant", messages[2].Role)
	s.Equal("assistant context", messages[2].Content)
	s.Equal("user", messages[3].Role)
	s.Equal("final prompt", messages[3].Content)
}

func (s *ContentSuite) TestBuildMessagesSkipsEmptyContent() {
	messages, contextCount, err := buildMessagesWithContext("prompt", []*model.PromptContext{
		{MessageType: model.ContextMessageTypeSystem, Content: "  "},
		nil,
		{MessageType: model.ContextMessageTypeHuman, Content: "valid"},
	})

	s.Require().NoError(err)
	s.Equal(1, contextCount)
	s.Len(messages, 2)
	s.Equal("user", messages[0].Role)
	s.Equal("valid", messages[0].Content)
}

func (s *ContentSuite) TestExtractJSONPayload() {
	text := "Here is JSON:\n```json\n{\"status\":\"ok\"}\n```"
	payload := extractJSONPayload(text)
	s.Equal("{\"status\":\"ok\"}", payload)
}

func (s *ContentSuite) TestExtractJSONPayloadPlainJSON() {
	text := "{\"key\": \"value\"}"
	payload := extractJSONPayload(text)
	s.Equal("{\"key\": \"value\"}", payload)
}

func (s *ContentSuite) TestExtractTextFromResponseNil() {
	s.Equal("", extractTextFromResponse(nil))
}

func (s *ContentSuite) TestExtractTextFromResponseEmpty() {
	s.Equal("", extractTextFromResponse(&chatCompletionResponse{}))
}

func (s *ContentSuite) TestExtractTextFromResponse() {
	response := &chatCompletionResponse{
		Choices: []chatCompletionChoice{
			{Message: chatMessage{Content: "  hello world  "}},
		},
	}
	s.Equal("hello world", extractTextFromResponse(response))
}

func (s *ContentSuite) TestEmptyPromptReturnsError() {
	_, err := NewStringContentGenerator("", model.WithAuthToken("tok"))
	s.Error(err)
	s.Contains(err.Error(), "prompt is required")
}

func (s *ContentSuite) TestMessagesWithContextProviderError() {
	g := &textGenerator{prompt: "hi"}
	g.AddPromptContextProvider(context.Background(), &stubPromptContextProvider{err: errors.New("provider failed")})

	_, _, err := g.messagesWithContext(context.Background(), "")
	s.Error(err)
	s.Contains(err.Error(), "provider failed")
}

func (s *ContentSuite) TestLastOutputReaderStructured() {
	g := &structuredGeneratorView[map[string]any]{
		structuredGenerator: &structuredGenerator[map[string]any]{
			LastOutput: "{\"ok\":true}",
		},
	}

	s.Equal("{\"ok\":true}", g.LastOutput())
}

func (s *ContentSuite) TestLastOutputReaderText() {
	g := &textGeneratorView{
		textGenerator: &textGenerator{
			LastOutput: "raw-text",
		},
	}

	s.Equal("raw-text", g.LastOutput())
}

func (s *ContentSuite) TestToolResultMessageMarshalsWithName() {
	msg := chatMessage{
		Role:       "tool",
		Name:       "get_secret_value",
		ToolCallID: "call_123",
		Content:    `{"secret":"value"}`,
	}

	body, err := json.Marshal(msg)

	s.Require().NoError(err)
	s.Contains(string(body), `"role":"tool"`)
	s.Contains(string(body), `"name":"get_secret_value"`)
	s.Contains(string(body), `"tool_call_id":"call_123"`)
	s.Contains(string(body), `"content":"{\"secret\":\"value\"}"`)
}

func (s *ContentSuite) TestChatCompletionRequestMarshalsToolChoiceAutoWhenSet() {
	req := chatCompletionRequest{
		Model:      "openai/gpt-oss-120b:cerebras",
		Messages:   []chatMessage{{Role: "user", Content: "hello"}},
		ToolChoice: "auto",
		Tools: []chatTool{
			{
				Type: "function",
				Function: chatFunction{
					Name:       "get_secret_value",
					Parameters: map[string]any{"type": "object"},
				},
			},
		},
	}

	body, err := json.Marshal(req)

	s.Require().NoError(err)
	s.Contains(string(body), `"tool_choice":"auto"`)
}

func (s *ContentSuite) TestChatCompletionRequestOmitsToolChoiceWhenUnset() {
	req := chatCompletionRequest{
		Model:    "openai/gpt-oss-120b:cerebras",
		Messages: []chatMessage{{Role: "user", Content: "hello"}},
	}

	body, err := json.Marshal(req)

	s.Require().NoError(err)
	s.NotContains(string(body), `"tool_choice"`)
}

type stubPromptContextProvider struct {
	err error
}

func (p *stubPromptContextProvider) GenerateContext(ctx context.Context) ([]*model.PromptContext, error) {
	if p.err != nil {
		return nil, p.err
	}
	return nil, nil
}
