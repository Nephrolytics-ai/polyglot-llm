package anthropic

import (
	"context"
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
	system, messages, contextCount, err := buildMessagesWithContext("final prompt", []*model.PromptContext{
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
	s.Equal("system one", system)
	s.Len(messages, 3)
	s.Equal("user", messages[0].Role)
	s.Equal("human context", messages[0].Content[0].Text)
	s.Equal("assistant", messages[1].Role)
	s.Equal("assistant context", messages[1].Content[0].Text)
	s.Equal("user", messages[2].Role)
	s.Equal("final prompt", messages[2].Content[0].Text)
}

func (s *ContentSuite) TestExtractJSONPayload() {
	text := "Here is JSON:\n```json\n{\"status\":\"ok\"}\n```"
	payload := extractJSONPayload(text)
	s.Equal("{\"status\":\"ok\"}", payload)
}

func (s *ContentSuite) TestMessagesWithContextProviderError() {
	g := &textGenerator{prompt: "hi"}
	g.AddPromptContextProvider(context.Background(), &stubPromptContextProvider{err: errors.New("provider failed")})

	_, _, _, err := g.messagesWithContext(context.Background(), "")
	s.Error(err)
	s.Contains(err.Error(), "provider failed")
}

type stubPromptContextProvider struct {
	err error
}

func (s *stubPromptContextProvider) GenerateContext(ctx context.Context) ([]*model.PromptContext, error) {
	if s.err != nil {
		return nil, s.err
	}
	return nil, nil
}
