package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/logging"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
)

type embeddingGenerator struct {
	client *client
	cfg    model.GeneratorConfig
}

func NewEmbeddingGenerator(opts ...model.GeneratorOption) (model.EmbeddingGenerator, error) {
	cfg := model.ResolveGeneratorOpts(opts...)
	c := newClient(cfg)
	return &embeddingGenerator{
		client: c,
		cfg:    cfg,
	}, nil
}

func (g *embeddingGenerator) Generate(
	ctx context.Context,
	input string,
) (model.EmbeddingVector, model.GenerationMetadata, error) {
	vectors, meta, err := g.GenerateBatch(ctx, []string{input})
	if err != nil {
		return nil, meta, utils.WrapIfNotNil(err)
	}
	if len(vectors) != 1 {
		return nil, meta, utils.WrapIfNotNil(
			fmt.Errorf("expected exactly 1 embedding vector, got %d", len(vectors)),
		)
	}
	return vectors[0], meta, nil
}

func (g *embeddingGenerator) GenerateBatch(
	ctx context.Context,
	inputs []string,
) (model.EmbeddingVectors, model.GenerationMetadata, error) {
	start := time.Now()
	modelName := resolveEmbeddingModelName(g.cfg)
	meta := initMetadata(modelName)
	defer setLatencyMetadata(meta, start)

	log := logging.NewLogger(ctx)
	err := validateEmbeddingInputs(inputs)
	if err != nil {
		log.Errorf("error: %v", err)
		return nil, meta, utils.WrapIfNotNil(err)
	}

	log.Infof(
		"embedding_request inputs=%d model=%q base_url=%q",
		len(inputs),
		modelName,
		g.client.baseURL,
	)

	vectors, err := g.client.embed(ctx, modelName, inputs)
	if err != nil {
		log.Errorf("error: %v", err)
		return nil, meta, utils.WrapIfNotNil(err)
	}

	meta[model.MetadataKeyEmbeddingCount] = fmt.Sprintf("%d", len(vectors))
	if len(vectors) > 0 {
		meta[model.MetadataKeyEmbeddingDims] = fmt.Sprintf("%d", len(vectors[0]))
	}
	meta[model.MetadataKeyOutputTokens] = "0"

	return vectors, meta, nil
}

type embedRequest struct {
	Model string `json:"model"`
	Input any    `json:"input"`
}

type embedResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

type legacyEmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type legacyEmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

func (c *client) embed(ctx context.Context, modelName string, inputs []string) (model.EmbeddingVectors, error) {
	reqBody, err := json.Marshal(embedRequest{
		Model: modelName,
		Input: inputs,
	})
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	httpReq, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		strings.TrimRight(c.baseURL, "/")+"/api/embed",
		bytes.NewReader(reqBody),
	)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 120 * time.Second}
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode >= 200 && httpResp.StatusCode < 300 {
		var resp embedResponse
		if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
			return nil, utils.WrapIfNotNil(err)
		}
		if len(resp.Embeddings) == 0 {
			return nil, utils.WrapIfNotNil(errors.New("embedding response has no data"))
		}
		if len(resp.Embeddings) != len(inputs) {
			return nil, utils.WrapIfNotNil(
				fmt.Errorf("embedding response size mismatch: expected %d, got %d", len(inputs), len(resp.Embeddings)),
			)
		}

		vectors := make(model.EmbeddingVectors, len(resp.Embeddings))
		for i, vec := range resp.Embeddings {
			vectors[i] = append(model.EmbeddingVector(nil), vec...)
		}
		return vectors, nil
	}

	// Backward compatibility fallback for older Ollama versions.
	if len(inputs) == 1 {
		legacyReqBody, err := json.Marshal(legacyEmbeddingRequest{
			Model:  modelName,
			Prompt: inputs[0],
		})
		if err != nil {
			return nil, utils.WrapIfNotNil(err)
		}

		legacyReq, err := http.NewRequestWithContext(
			ctx,
			http.MethodPost,
			strings.TrimRight(c.baseURL, "/")+"/api/embeddings",
			bytes.NewReader(legacyReqBody),
		)
		if err != nil {
			return nil, utils.WrapIfNotNil(err)
		}
		legacyReq.Header.Set("Content-Type", "application/json")

		legacyResp, err := httpClient.Do(legacyReq)
		if err != nil {
			return nil, utils.WrapIfNotNil(err)
		}
		defer legacyResp.Body.Close()

		if legacyResp.StatusCode >= 200 && legacyResp.StatusCode < 300 {
			var resp legacyEmbeddingResponse
			if err := json.NewDecoder(legacyResp.Body).Decode(&resp); err != nil {
				return nil, utils.WrapIfNotNil(err)
			}
			if len(resp.Embedding) == 0 {
				return nil, utils.WrapIfNotNil(errors.New("embedding response has no data"))
			}
			return model.EmbeddingVectors{
				append(model.EmbeddingVector(nil), resp.Embedding...),
			}, nil
		}
	}

	return nil, utils.WrapIfNotNil(fmt.Errorf("ollama embedding request failed with status %d", httpResp.StatusCode))
}

func validateEmbeddingInputs(inputs []string) error {
	if len(inputs) == 0 {
		return utils.WrapIfNotNil(errors.New("at least one input is required"))
	}

	for i, input := range inputs {
		if strings.TrimSpace(input) == "" {
			return utils.WrapIfNotNil(fmt.Errorf("input at index %d is empty", i))
		}
	}
	return nil
}
