package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"mime"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/logging"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
	"google.golang.org/genai"
)

type audioTranscriptionGenerator struct {
	filePath string
	opts     model.AudioOptions
	cfg      model.GeneratorConfig
}

func NewAudioTranscriptionGenerator(
	filePath string,
	opts model.AudioOptions,
) (model.AudioTranscriptionGenerator, error) {
	if strings.TrimSpace(filePath) == "" {
		return nil, utils.WrapIfNotNil(errors.New("file path is required"))
	}

	return &audioTranscriptionGenerator{
		filePath: filePath,
		opts:     cloneAudioOptions(opts),
		cfg:      audioGeneratorConfigFromOptions(opts),
	}, nil
}

func (g *audioTranscriptionGenerator) Generate(ctx context.Context) (string, model.GenerationMetadata, error) {
	start := time.Now()
	modelName := resolveAudioTranscriptionModelName(g.opts)
	meta := initMetadata(modelName)
	defer setLatencyMetadata(meta, start)

	log := logging.NewLogger(ctx)
	audioBytes, err := os.ReadFile(g.filePath)
	if err != nil {
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}

	mimeType, err := resolveAudioMIMEType(g.filePath)
	if err != nil {
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}

	client, err := newAPIClient(ctx, g.cfg)
	if err != nil {
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}

	prompt, err := buildAudioTranscriptionPrompt(g.opts)
	if err != nil {
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}
	contents := []*genai.Content{
		genai.NewContentFromParts(
			[]*genai.Part{
				genai.NewPartFromText(prompt),
				genai.NewPartFromBytes(audioBytes, mimeType),
			},
			genai.RoleUser,
		),
	}

	response, err := client.Models.GenerateContent(ctx, modelName, contents, &genai.GenerateContentConfig{})
	if err != nil {
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}

	transcript := strings.TrimSpace(response.Text())
	if transcript == "" {
		err = errors.New("transcription response is empty")
		log.Errorf("error: %v", err)
		return "", meta, utils.WrapIfNotNil(err)
	}

	applyAudioTranscriptionMetadata(meta, response)
	return transcript, meta, nil
}

func resolveAudioTranscriptionModelName(opts model.AudioOptions) string {
	if modelName := strings.TrimSpace(opts.Model); modelName != "" {
		return modelName
	}
	return defaultGenerationModelName
}

func audioGeneratorConfigFromOptions(opts model.AudioOptions) model.GeneratorConfig {
	cfg := model.GeneratorConfig{
		IgnoreInvalidGeneratorOptions: opts.IgnoreInvalidGeneratorOptions,
		URL:                           opts.URL,
		AuthToken:                     opts.AuthToken,
	}
	if modelName := strings.TrimSpace(opts.Model); modelName != "" {
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

func buildAudioTranscriptionPrompt(opts model.AudioOptions) (string, error) {
	customPrompt := strings.TrimSpace(opts.Prompt)
	if customPrompt != "" {
		return customPrompt, nil
	}

	base := "Transcribe this audio accurately. Return only the transcript text."
	keywordsPrompt, err := buildCommonMissedWordsPrompt(opts.Keywords)
	if err != nil {
		return "", err
	}
	if keywordsPrompt == "" {
		return base, nil
	}
	return base + "\n" + keywordsPrompt, nil
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

func resolveAudioMIMEType(filePath string) (string, error) {
	ext := strings.ToLower(filepath.Ext(strings.TrimSpace(filePath)))
	if ext == "" {
		return "", utils.WrapIfNotNil(errors.New("audio file extension is required to determine mime type"))
	}

	switch ext {
	case ".wav":
		return "audio/wav", nil
	case ".mp3":
		return "audio/mpeg", nil
	case ".m4a":
		return "audio/mp4", nil
	case ".mp4":
		return "audio/mp4", nil
	case ".webm":
		return "audio/webm", nil
	case ".ogg":
		return "audio/ogg", nil
	case ".flac":
		return "audio/flac", nil
	case ".aac":
		return "audio/aac", nil
	}

	mimeType := mime.TypeByExtension(ext)
	if mimeType == "" {
		return "", utils.WrapIfNotNil(errors.New("unsupported audio file extension: " + ext))
	}

	// Strip parameters such as "; charset=utf-8".
	mimeType = strings.TrimSpace(strings.Split(mimeType, ";")[0])
	if !strings.HasPrefix(mimeType, "audio/") {
		return "", utils.WrapIfNotNil(errors.New("unsupported audio mime type: " + mimeType))
	}
	return mimeType, nil
}

func applyAudioTranscriptionMetadata(meta model.GenerationMetadata, response *genai.GenerateContentResponse) {
	if meta == nil || response == nil || response.UsageMetadata == nil {
		return
	}

	meta[model.MetadataKeyInputTokens] = strconv.Itoa(int(response.UsageMetadata.PromptTokenCount))
	meta[model.MetadataKeyOutputTokens] = strconv.Itoa(int(response.UsageMetadata.CandidatesTokenCount))
	meta[model.MetadataKeyTotalTokens] = strconv.Itoa(int(response.UsageMetadata.TotalTokenCount))
	meta[model.MetadataKeyCachedInputTokens] = strconv.Itoa(int(response.UsageMetadata.CachedContentTokenCount))
	meta[model.MetadataKeyReasoningTokens] = strconv.Itoa(int(response.UsageMetadata.ThoughtsTokenCount))
	if strings.TrimSpace(response.ResponseID) != "" {
		meta[model.MetadataKeyResponseID] = response.ResponseID
	}
	if len(response.Candidates) > 0 && response.Candidates[0] != nil {
		meta[model.MetadataKeyResponseStatus] = string(response.Candidates[0].FinishReason)
	}
}
