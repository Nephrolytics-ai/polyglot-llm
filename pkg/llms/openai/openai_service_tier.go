package openai

import (
	"fmt"

	"github.com/Nephrolytics-ai/polyglot-llm/pkg/model"
	"github.com/Nephrolytics-ai/polyglot-llm/pkg/utils"
	"github.com/openai/openai-go/v3/responses"
)

const serviceTierProviderOptionKey = "openai.service_tier"

type ServiceTier string

const (
	ServiceTierStandard ServiceTier = "standard"
	ServiceTierAuto     ServiceTier = "auto"
	ServiceTierFlex     ServiceTier = "flex"
	ServiceTierFast     ServiceTier = "fast"
	ServiceTierPriority ServiceTier = "priority"
)

func WithServiceTier(value ServiceTier) model.GeneratorOption {
	return model.WithProviderOption(serviceTierProviderOptionKey, value)
}

func resolveServiceTier(cfg model.GeneratorConfig) (*responses.ResponseNewParamsServiceTier, error) {
	if len(cfg.ProviderOptions) == 0 {
		return nil, nil
	}

	raw, ok := cfg.ProviderOptions[serviceTierProviderOptionKey]
	if !ok || raw == nil {
		return nil, nil
	}

	value, ok := raw.(ServiceTier)
	if !ok {
		return nil, utils.WrapIfNotNil(fmt.Errorf("invalid openai service tier option type %T", raw))
	}

	var tier responses.ResponseNewParamsServiceTier
	switch value {
	case ServiceTierStandard:
		tier = responses.ResponseNewParamsServiceTierDefault
	case ServiceTierAuto:
		tier = responses.ResponseNewParamsServiceTierAuto
	case ServiceTierFlex:
		tier = responses.ResponseNewParamsServiceTierFlex
	case ServiceTierFast:
		tier = responses.ResponseNewParamsServiceTier("fast")
	case ServiceTierPriority:
		tier = responses.ResponseNewParamsServiceTierPriority
	default:
		return nil, utils.WrapIfNotNil(fmt.Errorf("unsupported openai service tier %q", value))
	}

	return &tier, nil
}
