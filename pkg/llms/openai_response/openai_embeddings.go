package openai_response

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/logging"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
	openai "github.com/openai/openai-go/v3"
)

const defaultEmbeddingModelName = "text-embedding-3-small"

type embeddingGenerator struct {
	client *client
	cfg    model.GeneratorConfig
}

func NewEmbeddingGenerator(opts ...model.GeneratorOption) (model.EmbeddingGenerator, error) {
	cfg := model.ResolveGeneratorOpts(opts...)
	c, err := newClient(cfg)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}
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
	meta := initMetadata(providerName, resolveEmbeddingModelName(g.cfg))
	defer setLatencyMetadata(meta, start)

	logging.NewLogger(ctx).Infof(
		"embedding_request inputs=%d model=%q dimensions=%v",
		len(inputs),
		resolveEmbeddingModelName(g.cfg),
		g.cfg.EmbeddingDimensions,
	)

	vectors, response, err := g.client.runEmbeddings(ctx, inputs, g.cfg)
	if err != nil {
		return nil, meta, utils.WrapIfNotNil(err)
	}
	applyOpenAIEmbeddingMetadata(meta, response, vectors)
	return vectors, meta, nil
}

func (c *client) runEmbeddings(
	ctx context.Context,
	inputs []string,
	cfg model.GeneratorConfig,
) (model.EmbeddingVectors, *openai.CreateEmbeddingResponse, error) {
	err := validateEmbeddingInputs(inputs)
	if err != nil {
		return nil, nil, utils.WrapIfNotNil(err)
	}
	if cfg.EmbeddingDimensions != nil && *cfg.EmbeddingDimensions <= 0 {
		return nil, nil, utils.WrapIfNotNil(errors.New("embedding dimensions must be greater than zero"))
	}

	params := openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: append([]string(nil), inputs...),
		},
		Model: openai.EmbeddingModel(resolveEmbeddingModelName(cfg)),
	}
	if cfg.EmbeddingDimensions != nil {
		params.Dimensions = openai.Int(int64(*cfg.EmbeddingDimensions))
	}

	response, err := c.apiClient.Embeddings.New(ctx, params)
	if err != nil {
		return nil, nil, utils.WrapIfNotNil(err)
	}
	if response == nil {
		return nil, nil, utils.WrapIfNotNil(errors.New("embeddings API returned nil response"))
	}

	vectors, err := convertEmbeddingResponse(response, len(inputs))
	if err != nil {
		return nil, nil, utils.WrapIfNotNil(err)
	}
	return vectors, response, nil
}

func resolveEmbeddingModelName(cfg model.GeneratorConfig) string {
	if cfg.Model != nil {
		modelName := strings.TrimSpace(*cfg.Model)
		if modelName != "" {
			return modelName
		}
	}
	return defaultEmbeddingModelName
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

func convertEmbeddingResponse(
	response *openai.CreateEmbeddingResponse,
	expected int,
) (model.EmbeddingVectors, error) {
	if response == nil {
		return nil, utils.WrapIfNotNil(errors.New("nil embedding response"))
	}
	if len(response.Data) == 0 {
		return nil, utils.WrapIfNotNil(errors.New("embedding response has no data"))
	}
	if expected > 0 && len(response.Data) != expected {
		return nil, utils.WrapIfNotNil(
			fmt.Errorf("embedding response size mismatch: expected %d, got %d", expected, len(response.Data)),
		)
	}

	vectors := make(model.EmbeddingVectors, len(response.Data))
	seen := make(map[int]struct{}, len(response.Data))
	for _, item := range response.Data {
		idx := int(item.Index)
		if idx < 0 || idx >= len(vectors) {
			return nil, utils.WrapIfNotNil(
				fmt.Errorf("embedding index out of range: %d", item.Index),
			)
		}
		if _, exists := seen[idx]; exists {
			return nil, utils.WrapIfNotNil(
				fmt.Errorf("duplicate embedding index: %d", item.Index),
			)
		}
		seen[idx] = struct{}{}

		vector := make(model.EmbeddingVector, len(item.Embedding))
		for i, value := range item.Embedding {
			vector[i] = value
		}
		vectors[idx] = vector
	}

	for i, vector := range vectors {
		if vector == nil {
			return nil, utils.WrapIfNotNil(
				fmt.Errorf("missing embedding vector for index %d", i),
			)
		}
	}
	return vectors, nil
}

func applyOpenAIEmbeddingMetadata(
	meta model.GenerationMetadata,
	response *openai.CreateEmbeddingResponse,
	vectors model.EmbeddingVectors,
) {
	if meta == nil {
		return
	}

	meta[model.MetadataKeyEmbeddingCount] = strconv.Itoa(len(vectors))
	if len(vectors) > 0 {
		meta[model.MetadataKeyEmbeddingDims] = strconv.Itoa(len(vectors[0]))
	}

	if response == nil {
		return
	}

	if strings.TrimSpace(response.Model) != "" {
		meta[model.MetadataKeyModel] = response.Model
	}
	meta[model.MetadataKeyInputTokens] = strconv.FormatInt(response.Usage.PromptTokens, 10)
	meta[model.MetadataKeyTotalTokens] = strconv.FormatInt(response.Usage.TotalTokens, 10)
	meta[model.MetadataKeyOutputTokens] = "0"
}
