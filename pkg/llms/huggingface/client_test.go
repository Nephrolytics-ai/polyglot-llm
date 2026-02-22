package huggingface

import (
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

func (s *ClientSuite) TestResolveModelNameFromConfig() {
	name := "custom-model"
	cfg := model.GeneratorConfig{Model: &name}
	s.Equal("custom-model", resolveModelName(cfg))
}

func (s *ClientSuite) TestResolveModelNameDefault() {
	cfg := model.GeneratorConfig{}
	s.Equal(defaultModelName, resolveModelName(cfg))
}

func (s *ClientSuite) TestResolveEmbeddingModelNameFromConfig() {
	name := "custom-embed-model"
	cfg := model.GeneratorConfig{Model: &name}
	s.Equal("custom-embed-model", resolveEmbeddingModelName(cfg))
}

func (s *ClientSuite) TestResolveEmbeddingModelNameDefault() {
	cfg := model.GeneratorConfig{}
	s.Equal(defaultEmbeddingModelName, resolveEmbeddingModelName(cfg))
}

func (s *ClientSuite) TestResolveMaxTokensFromConfig() {
	maxTokens := 512
	cfg := model.GeneratorConfig{MaxTokens: &maxTokens}
	s.Equal(512, resolveMaxTokens(cfg))
}

func (s *ClientSuite) TestResolveMaxTokensDefault() {
	cfg := model.GeneratorConfig{}
	s.Equal(defaultMaxTokens, resolveMaxTokens(cfg))
}

func (s *ClientSuite) TestNewAPIClientRequiresAuthToken() {
	cfg := model.GeneratorConfig{}
	client, err := newAPIClient(cfg)
	s.Nil(client)
	s.Error(err)
	s.Contains(err.Error(), "auth token is required")
}

func (s *ClientSuite) TestNewAPIClientSuccess() {
	cfg := model.GeneratorConfig{AuthToken: "hf_test_token"}
	client, err := newAPIClient(cfg)
	s.NoError(err)
	s.NotNil(client)
	s.Equal("hf_test_token", client.apiKey)
	s.Equal(defaultBaseURL, client.baseURL)
}

func (s *ClientSuite) TestNewAPIClientCustomBaseURL() {
	cfg := model.GeneratorConfig{
		AuthToken: "hf_test_token",
		URL:       "https://custom-hf.example.com/",
	}
	client, err := newAPIClient(cfg)
	s.NoError(err)
	s.Equal("https://custom-hf.example.com", client.baseURL)
}

func (s *ClientSuite) TestInitMetadata() {
	meta := initMetadata("test-model")
	s.Equal(providerName, meta[model.MetadataKeyProvider])
	s.Equal("test-model", meta[model.MetadataKeyModel])
}

func (s *ClientSuite) TestInitMetadataEmptyModel() {
	meta := initMetadata("")
	s.Equal("unknown", meta[model.MetadataKeyModel])
}

func (s *ClientSuite) TestAccumulateUsageTotalsNilSafe() {
	accumulateUsageTotals(nil, nil)
	accumulateUsageTotals(&flowUsageTotals{}, nil)
}

func (s *ClientSuite) TestAccumulateUsageTotals() {
	totals := &flowUsageTotals{}
	response := &chatCompletionResponse{
		Usage: &chatCompletionUsage{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		},
	}
	accumulateUsageTotals(totals, response)
	s.Equal(1, totals.APICalls)
	s.Equal(int64(100), totals.InputTokens)
	s.Equal(int64(50), totals.OutputTokens)
	s.Equal(int64(150), totals.TotalTokens)
}
