package gemini

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/logging"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
	"google.golang.org/genai"
)

type embeddingGenerator struct {
	cfg model.GeneratorConfig
}

func NewEmbeddingGenerator(opts ...model.GeneratorOption) (model.EmbeddingGenerator, error) {
	cfg := model.ResolveGeneratorOpts(opts...)
	return &embeddingGenerator{
		cfg: cfg,
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
		return nil, meta, utils.WrapIfNotNil(fmt.Errorf("expected exactly 1 embedding vector, got %d", len(vectors)))
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

	if g.cfg.EmbeddingDimensions != nil && *g.cfg.EmbeddingDimensions <= 0 {
		err = errors.New("embedding dimensions must be greater than zero")
		log.Errorf("error: %v", err)
		return nil, meta, utils.WrapIfNotNil(err)
	}

	client, err := newAPIClient(ctx, g.cfg)
	if err != nil {
		log.Errorf("error: %v", err)
		return nil, meta, utils.WrapIfNotNil(err)
	}

	contents := make([]*genai.Content, 0, len(inputs))
	for _, input := range inputs {
		contents = append(contents, genai.NewContentFromText(input, genai.RoleUser))
	}

	config := &genai.EmbedContentConfig{}
	if g.cfg.EmbeddingDimensions != nil {
		dims := int32(*g.cfg.EmbeddingDimensions)
		config.OutputDimensionality = &dims
	}

	log.Infof(
		"embedding_request inputs=%d model=%q dimensions=%v",
		len(inputs),
		modelName,
		g.cfg.EmbeddingDimensions,
	)

	response, err := client.Models.EmbedContent(ctx, modelName, contents, config)
	if err != nil {
		log.Errorf("error: %v", err)
		return nil, meta, utils.WrapIfNotNil(err)
	}

	vectors, err := convertEmbeddingResponse(response, len(inputs))
	if err != nil {
		log.Errorf("error: %v", err)
		return nil, meta, utils.WrapIfNotNil(err)
	}

	applyEmbeddingMetadata(meta, vectors)
	return vectors, meta, nil
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
	response *genai.EmbedContentResponse,
	expected int,
) (model.EmbeddingVectors, error) {
	if response == nil {
		return nil, utils.WrapIfNotNil(errors.New("nil embedding response"))
	}
	if len(response.Embeddings) == 0 {
		return nil, utils.WrapIfNotNil(errors.New("embedding response has no data"))
	}
	if expected > 0 && len(response.Embeddings) != expected {
		return nil, utils.WrapIfNotNil(
			fmt.Errorf("embedding response size mismatch: expected %d, got %d", expected, len(response.Embeddings)),
		)
	}

	vectors := make(model.EmbeddingVectors, len(response.Embeddings))
	for i, embedding := range response.Embeddings {
		if embedding == nil {
			return nil, utils.WrapIfNotNil(fmt.Errorf("missing embedding at index %d", i))
		}
		vector := make(model.EmbeddingVector, len(embedding.Values))
		for j, value := range embedding.Values {
			vector[j] = float64(value)
		}
		vectors[i] = vector
	}
	return vectors, nil
}
