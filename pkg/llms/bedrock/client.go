package bedrock

import (
	"context"
	"errors"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

const (
	defaultModelName = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
	maxToolRounds    = 12
	providerName     = "bedrock"
	defaultRegion    = "us-east-1"
)

type flowUsageTotals struct {
	APICalls          int
	ToolRounds        int
	InputTokens       int64
	OutputTokens      int64
	TotalTokens       int64
	CachedInputTokens int64
}

func newClient(ctx context.Context, cfg model.GeneratorConfig) (*bedrockruntime.Client, error) {
	awsCfg, err := loadAWSConfig(ctx)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	client := bedrockruntime.NewFromConfig(awsCfg, func(o *bedrockruntime.Options) {
		if strings.TrimSpace(cfg.URL) != "" {
			o.BaseEndpoint = aws.String(strings.TrimSpace(cfg.URL))
		}
	})
	return client, nil
}

func loadAWSConfig(ctx context.Context) (aws.Config, error) {
	region := strings.TrimSpace(os.Getenv("AWS_REGION"))
	if region == "" {
		region = defaultRegion
	}

	accessKeyID := strings.TrimSpace(os.Getenv("AWS_ACCESS_KEY_ID"))
	secretAccessKey := strings.TrimSpace(os.Getenv("AWS_SECRET_ACCESS_KEY"))
	profile := strings.TrimSpace(os.Getenv("AWS_PROFILE"))

	loadOpts := []func(*config.LoadOptions) error{
		config.WithRegion(region),
	}

	switch {
	case accessKeyID != "" || secretAccessKey != "":
		if accessKeyID == "" || secretAccessKey == "" {
			return aws.Config{}, utils.WrapIfNotNil(
				errors.New("both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required when using key-based auth"),
			)
		}

		sessionToken := strings.TrimSpace(os.Getenv("AWS_SESSION_TOKEN"))
		loadOpts = append(loadOpts, config.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider(accessKeyID, secretAccessKey, sessionToken),
		))
	case profile != "":
		loadOpts = append(loadOpts, config.WithSharedConfigProfile(profile))
	default:
		return aws.Config{}, utils.WrapIfNotNil(
			errors.New("missing AWS credentials: set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE"),
		)
	}

	cfg, err := config.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		return aws.Config{}, utils.WrapIfNotNil(err)
	}
	return cfg, nil
}

func resolveModelName(cfg model.GeneratorConfig) string {
	if cfg.Model != nil {
		modelName := strings.TrimSpace(*cfg.Model)
		if modelName != "" {
			return modelName
		}
	}
	return defaultModelName
}

func initMetadata(modelName string) model.GenerationMetadata {
	if strings.TrimSpace(modelName) == "" {
		modelName = "unknown"
	}

	return model.GenerationMetadata{
		model.MetadataKeyProvider: providerName,
		model.MetadataKeyModel:    modelName,
	}
}

func setLatencyMetadata(meta model.GenerationMetadata, start time.Time) {
	if meta == nil {
		return
	}
	meta[model.MetadataKeyLatencyMs] = strconv.FormatInt(time.Since(start).Milliseconds(), 10)
}

func applyBedrockMetadata(
	meta model.GenerationMetadata,
	totals flowUsageTotals,
	stopReason string,
	responseLatencyMs int64,
) {
	if meta == nil {
		return
	}

	meta[model.MetadataKeyAPICalls] = strconv.Itoa(totals.APICalls)
	meta[model.MetadataKeyToolRounds] = strconv.Itoa(totals.ToolRounds)
	meta[model.MetadataKeyInputTokens] = strconv.FormatInt(totals.InputTokens, 10)
	meta[model.MetadataKeyOutputTokens] = strconv.FormatInt(totals.OutputTokens, 10)
	meta[model.MetadataKeyTotalTokens] = strconv.FormatInt(totals.TotalTokens, 10)
	meta[model.MetadataKeyCachedInputTokens] = strconv.FormatInt(totals.CachedInputTokens, 10)
	meta[model.MetadataKeyReasoningTokens] = "0"

	if strings.TrimSpace(stopReason) != "" {
		meta[model.MetadataKeyResponseStatus] = stopReason
	}
	if responseLatencyMs > 0 {
		meta[model.MetadataKeyLatencyMs] = strconv.FormatInt(responseLatencyMs, 10)
	}
}
