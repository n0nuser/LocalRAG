"""Cost estimation utilities.

Prices are approximate USD per 1 000 tokens (input+output blended).
Update ``PRICE_PER_1K_TOKENS`` when provider pricing changes.
"""

from __future__ import annotations

PRICE_PER_1K_TOKENS: dict[str, float] = {
    # OpenAI
    "gpt-4o": 0.005,
    "gpt-4o-mini": 0.00015,
    "gpt-4-turbo": 0.01,
    "gpt-3.5-turbo": 0.0005,
    # Anthropic
    "claude-opus-4-5": 0.015,
    "claude-sonnet-4-5": 0.003,
    "claude-haiku-4-5": 0.00025,
    # Ollama (local — free)
    "_default_ollama": 0.0,
}

_FALLBACK_PRICE = 0.002


def estimate_cost_usd(model: str, tokens: int) -> float:
    """Return estimated USD cost for ``tokens`` tokens on ``model``.

    Uses an exact match first, then a prefix match (e.g. ``gpt-4o-mini-2024``
    matches ``gpt-4o-mini``), then the generic fallback.
    Returns 0.0 for local/Ollama models.
    """
    if tokens <= 0:
        return 0.0

    price = PRICE_PER_1K_TOKENS.get(model)
    if price is None:
        for key in sorted(PRICE_PER_1K_TOKENS, key=len, reverse=True):
            if model.startswith(key):
                price = PRICE_PER_1K_TOKENS[key]
                break
    if price is None:
        price = _FALLBACK_PRICE

    return round(price * tokens / 1000, 8)
