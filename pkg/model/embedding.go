package model

type EmbeddingVector = []float64
type EmbeddingVectors = [][]float64

const (
	MetadataKeyEmbeddingCount = "embedding_count"
	MetadataKeyEmbeddingDims  = "embedding_dims"
)

func WithEmbeddingDimensions(value int) GeneratorOption {
	return generatorOptionFunc(func(cfg *GeneratorConfig) {
		cfg.EmbeddingDimensions = &value
	})
}
