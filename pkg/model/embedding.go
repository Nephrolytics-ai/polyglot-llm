package model

import "context"

// NewEmbeddingGeneratorFunc creates an embedding generator.
// Inputs are provided at Generate / GenerateBatch call time.
type NewEmbeddingGeneratorFunc func(opts ...GeneratorOption) (EmbeddingGenerator, error)

type EmbeddingVector = []float64
type EmbeddingVectors = [][]float64

type EmbeddingGenerator interface {
	Generate(ctx context.Context, input string) (EmbeddingVector, GenerationMetadata, error)
	GenerateBatch(ctx context.Context, inputs []string) (EmbeddingVectors, GenerationMetadata, error)
}

const (
	MetadataKeyEmbeddingCount = "embedding_count"
	MetadataKeyEmbeddingDims  = "embedding_dims"
)

func WithEmbeddingDimensions(value int) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.EmbeddingDimensions = &value
	})
}
