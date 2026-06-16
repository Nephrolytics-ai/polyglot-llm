package openai

import (
	"fmt"
	"strings"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
)

// Supported OpenAI provider option keys:
//   - openai.prompt_cache_key
const promptCacheKeyProviderOptionKey = "openai.prompt_cache_key"

func resolvePromptCacheKey(cfg model.GeneratorConfig) (*string, error) {
	if len(cfg.ProviderOptions) == 0 {
		return nil, nil
	}

	raw, ok := cfg.ProviderOptions[promptCacheKeyProviderOptionKey]
	if !ok || raw == nil {
		return nil, nil
	}

	value, ok := raw.(string)
	if !ok {
		return nil, utils.WrapIfNotNil(fmt.Errorf("invalid openai prompt cache key option type %T", raw))
	}

	value = strings.TrimSpace(value)
	if value == "" {
		return nil, nil
	}

	return &value, nil
}
