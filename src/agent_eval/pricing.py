from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Per-million-token USD pricing table for LLM cost estimation."""

    model_name: str
    input_per_1m_usd: float
    output_per_1m_usd: float
    cached_input_per_1m_usd: float = 0.0


@dataclass(frozen=True)
class CostBreakdown:
    """Computed token cost breakdown for a single agent trace."""

    model_name: str
    input_cost_usd: float
    cached_input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


DEFAULT_MODEL_PRICING: dict[str, ModelPricing] = {
    "gemini-2.5-pro": ModelPricing("gemini-2.5-pro", 1.25, 10.00, 0.31),
    "gemini-2.5-flash": ModelPricing("gemini-2.5-flash", 0.30, 2.50, 0.075),
    "claude-sonnet-4": ModelPricing("claude-sonnet-4", 3.00, 15.00, 0.30),
    "claude-haiku-3.5": ModelPricing("claude-haiku-3.5", 0.80, 4.00, 0.08),
    "gpt-4o": ModelPricing("gpt-4o", 2.50, 10.00, 1.25),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", 0.15, 0.60, 0.075),
    "llama-3.3-70b-local": ModelPricing("llama-3.3-70b-local", 0.0, 0.0, 0.0),
}

FALLBACK_PRICING = ModelPricing("unknown-default", 1.00, 3.00, 0.25)


def estimate_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
    pricing_registry: dict[str, ModelPricing] | None = None,
) -> CostBreakdown:
    registry = pricing_registry or DEFAULT_MODEL_PRICING
    pricing = registry.get(model_name, FALLBACK_PRICING)

    billable_cached = min(max(cached_prompt_tokens, 0), max(prompt_tokens, 0))
    billable_uncached = max(prompt_tokens - billable_cached, 0)
    billable_output = max(completion_tokens, 0)

    input_cost = (billable_uncached / 1_000_000) * pricing.input_per_1m_usd
    cached_cost = (billable_cached / 1_000_000) * pricing.cached_input_per_1m_usd
    output_cost = (billable_output / 1_000_000) * pricing.output_per_1m_usd
    total = input_cost + cached_cost + output_cost

    return CostBreakdown(
        model_name=model_name,
        input_cost_usd=round(input_cost, 6),
        cached_input_cost_usd=round(cached_cost, 6),
        output_cost_usd=round(output_cost, 6),
        total_cost_usd=round(total, 6),
    )
