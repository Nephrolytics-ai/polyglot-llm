package openai

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/logging"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
)

const defaultAudioTranscriptionModelName = "whisper-1"

type audioTranscriptionGenerator struct {
	client   *client
	filePath string
	opts     model.AudioOptions
}

func NewAudioTranscriptionGenerator(
	filePath string,
	opts model.AudioOptions,
) (model.AudioTranscriptionGenerator, error) {
	if strings.TrimSpace(filePath) == "" {
		return nil, utils.WrapIfNotNil(errors.New("file path is required"))
	}

	cfg := audioGeneratorConfigFromOptions(opts)
	c, err := newClient(cfg)
	if err != nil {
		return nil, utils.WrapIfNotNil(err)
	}

	return &audioTranscriptionGenerator{
		client:   c,
		filePath: filePath,
		opts:     cloneAudioOptions(opts),
	}, nil
}

func (g *audioTranscriptionGenerator) Generate(ctx context.Context) (string, model.GenerationMetadata, error) {
	start := time.Now()
	meta := initMetadata(providerName, resolveAudioTranscriptionModelName(g.opts))
	defer setLatencyMetadata(meta, start)

	logging.NewLogger(ctx).Infof(
		"audio_transcription_request model=%q",
		resolveAudioTranscriptionModelName(g.opts),
	)

	transcript, response, err := g.client.runAudioTranscription(ctx, g.filePath, g.opts)
	if err != nil {
		logging.NewLogger(ctx).Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}

	applyOpenAIAudioTranscriptionMetadata(meta, response)
	return transcript, meta, nil
}

func (c *client) runAudioTranscription(
	ctx context.Context,
	filePath string,
	opts model.AudioOptions,
) (string, *openai.AudioTranscriptionNewResponseUnion, error) {
	if strings.TrimSpace(filePath) == "" {
		return "", nil, utils.WrapIfNotNil(errors.New("file path is required"))
	}

	file, err := os.Open(filePath)
	if err != nil {
		return "", nil, utils.WrapIfNotNil(err)
	}
	defer func() {
		_ = file.Close()
	}()

	params := openai.AudioTranscriptionNewParams{
		File:           file,
		Model:          openai.AudioModel(resolveAudioTranscriptionModelName(opts)),
		ResponseFormat: openai.AudioResponseFormatJSON,
	}
	prompt, err := buildAudioTranscriptionPrompt(opts)
	if err != nil {
		return "", nil, utils.WrapIfNotNil(err)
	}
	if prompt != "" {
		params.Prompt = param.NewOpt(prompt)
	}

	response, err := c.apiClient.Audio.Transcriptions.New(ctx, params)
	if err != nil {
		return "", nil, utils.WrapIfNotNil(err)
	}
	if response == nil {
		return "", nil, utils.WrapIfNotNil(errors.New("audio transcriptions API returned nil response"))
	}

	transcript := strings.TrimSpace(response.Text)
	if transcript == "" {
		return "", response, utils.WrapIfNotNil(errors.New("transcription response is empty"))
	}

	return transcript, response, nil
}

func buildAudioTranscriptionPrompt(opts model.AudioOptions) (string, error) {
	customPrompt := strings.TrimSpace(opts.Prompt)
	if customPrompt != "" {
		return customPrompt, nil
	}

	return buildCommonMissedWordsPrompt(opts.Keywords)
}

func buildCommonMissedWordsPrompt(keywords []model.AudioKeyword) (string, error) {
	normalizedKeywords := normalizeAudioKeywords(keywords)
	if len(normalizedKeywords) == 0 {
		return "", nil
	}

	keywordsJSON, err := json.Marshal(normalizedKeywords)
	if err != nil {
		return "", err
	}

	return "Common missed words: " + string(keywordsJSON), nil
}

func normalizeAudioKeywords(keywords []model.AudioKeyword) []model.AudioKeyword {
	if len(keywords) == 0 {
		return nil
	}

	normalized := make([]model.AudioKeyword, 0, len(keywords))
	for _, keyword := range keywords {
		word := strings.TrimSpace(keyword.Word)
		definition := strings.TrimSpace(keyword.Definition)
		commonMistypes := make([]string, 0, len(keyword.CommonMistypes))
		for _, candidate := range keyword.CommonMistypes {
			candidate = strings.TrimSpace(candidate)
			if candidate == "" {
				continue
			}
			commonMistypes = append(commonMistypes, candidate)
		}

		if word == "" && definition == "" && len(commonMistypes) == 0 {
			continue
		}

		normalized = append(normalized, model.AudioKeyword{
			Word:           word,
			CommonMistypes: commonMistypes,
			Definition:     definition,
		})
	}

	if len(normalized) == 0 {
		return nil
	}

	return normalized
}

func resolveAudioTranscriptionModelName(opts model.AudioOptions) string {
	modelName := strings.TrimSpace(opts.Model)
	if modelName != "" {
		return modelName
	}

	return defaultAudioTranscriptionModelName
}

func audioGeneratorConfigFromOptions(opts model.AudioOptions) model.GeneratorConfig {
	cfg := model.GeneratorConfig{
		IgnoreInvalidGeneratorOptions: opts.IgnoreInvalidGeneratorOptions,
		URL:                           opts.URL,
		AuthToken:                     opts.AuthToken,
	}

	modelName := strings.TrimSpace(opts.Model)
	if modelName != "" {
		cfg.Model = &modelName
	}

	return cfg
}

func cloneAudioOptions(opts model.AudioOptions) model.AudioOptions {
	cloned := opts
	if len(opts.Keywords) == 0 {
		cloned.Keywords = nil
		return cloned
	}

	cloned.Keywords = make([]model.AudioKeyword, len(opts.Keywords))
	for i, keyword := range opts.Keywords {
		clonedKeyword := keyword
		if len(keyword.CommonMistypes) > 0 {
			clonedKeyword.CommonMistypes = append([]string(nil), keyword.CommonMistypes...)
		} else {
			clonedKeyword.CommonMistypes = nil
		}
		cloned.Keywords[i] = clonedKeyword
	}

	return cloned
}

func applyOpenAIAudioTranscriptionMetadata(
	meta model.GenerationMetadata,
	response *openai.AudioTranscriptionNewResponseUnion,
) {
	if meta == nil || response == nil {
		return
	}

	meta[model.MetadataKeyInputTokens] = strconv.FormatInt(response.Usage.InputTokens, 10)
	meta[model.MetadataKeyOutputTokens] = strconv.FormatInt(response.Usage.OutputTokens, 10)
	meta[model.MetadataKeyTotalTokens] = strconv.FormatInt(response.Usage.TotalTokens, 10)
}
