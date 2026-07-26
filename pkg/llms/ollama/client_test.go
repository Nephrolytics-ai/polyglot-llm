package ollama

import (
	"net/http"
	"testing"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/stretchr/testify/suite"
)

type ClientSuite struct {
	suite.Suite
}

func TestClientSuite(t *testing.T) {
	suite.Run(t, new(ClientSuite))
}

func (s *ClientSuite) TestNewClientUsesAuthTokenFromConfig() {
	c := newClient(model.ResolveGeneratorOpts(
		model.WithURL("https://ollama.example.com"),
		model.WithAuthToken("token-123"),
	))

	s.Equal("https://ollama.example.com", c.baseURL)
	s.Equal("token-123", c.authToken)
}

func (s *ClientSuite) TestApplyAuthHeaderAddsBearerPrefix() {
	c := &client{authToken: "token-123"}
	headers := http.Header{}

	c.applyAuthHeader(headers)

	s.Equal("Bearer token-123", headers.Get("Authorization"))
}

func (s *ClientSuite) TestApplyAuthHeaderPreservesExistingBearerPrefix() {
	c := &client{authToken: "Bearer token-123"}
	headers := http.Header{}

	c.applyAuthHeader(headers)

	s.Equal("Bearer token-123", headers.Get("Authorization"))
}

func (s *ClientSuite) TestApplyAuthHeaderNoopsWhenTokenMissing() {
	c := &client{}
	headers := http.Header{}

	c.applyAuthHeader(headers)

	s.Equal("", headers.Get("Authorization"))
}
