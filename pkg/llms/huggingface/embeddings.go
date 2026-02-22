package huggingface

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/logging"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
)

type embeddingGenerator struct {
	client *apiClient
	cfg    model.GeneratorConfig
}

// featureExtractionRequest is the native HF Inference API request format.
type featureExtractionRequest struct {
	Inputs  []string                  `json:"inputs"`
	Options *featureExtractionOptions `json:"options,omitempty"`
}

type featureExtractionOptions struct {
	WaitForModel bool `json:"wait_for_model,omitempty"`
}

func NewEmbeddingGenerator(opts ...model.GeneratorOption) (model.EmbeddingGenerator, error) {
	cfg := model.ResolveGeneratorOpts(opts...)
	client, err := newEmbeddingAPIClient(cfg)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}
	return &embeddingGenerator{
		client: client,
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

	vectors, err := g.client.featureExtraction(ctx, modelName, inputs)
	if err != nil {
		log.Errorf("error: %v", err)
		return nil, meta, utils.WrapIfNotNil(err)
	}

	if len(vectors) == 0 {
		return nil, meta, utils.WrapIfNotNil(errors.New("embedding response has no data"))
	}
	if len(vectors) != len(inputs) {
		return nil, meta, utils.WrapIfNotNil(
			fmt.Errorf("embedding response size mismatch: expected %d, got %d", len(inputs), len(vectors)),
		)
	}

	meta[model.MetadataKeyEmbeddingCount] = fmt.Sprintf("%d", len(vectors))
	if len(vectors) > 0 {
		meta[model.MetadataKeyEmbeddingDims] = fmt.Sprintf("%d", len(vectors[0]))
	}
	meta[model.MetadataKeyOutputTokens] = "0"

	return vectors, meta, nil
}

// featureExtraction calls the native HF Inference API for embeddings.
// Endpoint: POST {baseURL}/models/{modelName}
// Request:  {"inputs": ["text1", "text2"], "options": {"wait_for_model": true}}
// Response for single input:  [0.1, 0.2, ...]  (1D array)
// Response for multiple inputs: [[0.1, 0.2, ...], [0.3, 0.4, ...]]  (2D array)
func (c *apiClient) featureExtraction(ctx context.Context, modelName string, inputs []string) (model.EmbeddingVectors, error) {
	request := featureExtractionRequest{
		Inputs:  inputs,
		Options: &featureExtractionOptions{WaitForModel: true},
	}

	requestBits, err := json.Marshal(request)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	endpoint := c.baseURL + "/models/" + modelName

	httpRequest, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		endpoint,
		bytes.NewReader(requestBits),
	)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Authorization", "Bearer "+c.apiKey)

	httpResponse, err := c.httpClient.Do(httpRequest)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}
	defer httpResponse.Body.Close()

	responseBits, err := io.ReadAll(httpResponse.Body)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	if httpResponse.StatusCode < 200 || httpResponse.StatusCode >= 300 {
		message := strings.TrimSpace(string(responseBits))
		apiErr := struct {
			Error string `json:"error"`
		}{}
		if unmarshalErr := json.Unmarshal(responseBits, &apiErr); unmarshalErr == nil {
			candidate := strings.TrimSpace(apiErr.Error)
			if candidate != "" {
				message = candidate
			}
		}
		if message == "" {
			message = "unknown huggingface embedding error"
		}
		return nil, utils.WrapIfNotNil(fmt.Errorf("huggingface embedding API error (%d): %s", httpResponse.StatusCode, message))
	}

	return parseFeatureExtractionResponse(responseBits, len(inputs))
}

// parseFeatureExtractionResponse handles the native HF response format.
// Single input returns a 1D array: [float64...]
// Multiple inputs return a 2D array: [[float64...]...]
// Token-level models may return a 3D array: [[[float64...]...]...]
func parseFeatureExtractionResponse(data []byte, expectedCount int) (model.EmbeddingVectors, error) {
	// Single input: try 1D array first.
	if expectedCount == 1 {
		var vector1D []float64
		if err := json.Unmarshal(data, &vector1D); err == nil && len(vector1D) > 0 {
			return model.EmbeddingVectors{vector1D}, nil
		}
	}

	// Multiple inputs: try 2D array (sentence-level embeddings).
	var vectors2D [][]float64
	if err := json.Unmarshal(data, &vectors2D); err == nil && len(vectors2D) > 0 {
		result := make(model.EmbeddingVectors, len(vectors2D))
		for i, vec := range vectors2D {
			result[i] = append(model.EmbeddingVector(nil), vec...)
		}
		return result, nil
	}

	// Fallback: 3D array (token-level embeddings). Mean-pool to get sentence vectors.
	var vectors3D [][][]float64
	if err := json.Unmarshal(data, &vectors3D); err == nil && len(vectors3D) > 0 {
		result := make(model.EmbeddingVectors, len(vectors3D))
		for i, tokenVectors := range vectors3D {
			result[i] = meanPool(tokenVectors)
		}
		return result, nil
	}

	return nil, utils.WrapIfNotNil(errors.New("unable to parse huggingface embedding response"))
}

// meanPool averages token-level embeddings into a single sentence vector.
func meanPool(tokenVectors [][]float64) model.EmbeddingVector {
	if len(tokenVectors) == 0 {
		return nil
	}

	dims := len(tokenVectors[0])
	result := make(model.EmbeddingVector, dims)
	count := float64(len(tokenVectors))

	for _, vec := range tokenVectors {
		for j, v := range vec {
			if j < dims {
				result[j] += v
			}
		}
	}
	for j := range result {
		result[j] /= count
	}
	return result
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
