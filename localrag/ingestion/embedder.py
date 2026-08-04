from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from localrag.embedding.base import (
    EmbeddingResponseError,
    EmbeddingTransportError,
    validate_vectors,
)
from localrag.ollama.schemas import (
    OllamaEmbedRequest,
    OllamaEmbedResponse,
    parse_ollama_json,
)

logger = logging.getLogger(__name__)


@dataclass
class OllamaEmbedder:
    base_url: str
    model: str
    timeout_seconds: float = 120.0
    provider_name: str = "ollama"
    dimension: int | None = None

    def embed(self, text: str, *, model: str | None = None) -> list[float]:
        rows = self.embed_batch([text], model=model)
        return rows[0]

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        model: str | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        if batch_size is not None and batch_size < 1:
            raise EmbeddingResponseError("batch_size must be positive")
        safe_batch_size = batch_size or len(texts)
        out: list[list[float]] = []
        logger.debug(
            "ollama_embed_batch total_texts=%s batch_size=%s",
            len(texts),
            safe_batch_size,
        )
        for start in range(0, len(texts), safe_batch_size):
            batch = texts[start : start + safe_batch_size]
            out.extend(self._embed_inputs(list(batch), model=model))
        return out

    # Compatibility aliases for callers from before the provider contract.
    def embed_text(self, text: str, *, model: str | None = None) -> list[float]:
        return self.embed(text, model=model)

    def embed_texts(
        self, texts: list[str], batch_size: int, *, model: str | None = None
    ) -> list[list[float]]:
        return self.embed_batch(texts, batch_size=batch_size, model=model)

    def _embed_inputs(self, inputs: list[str], *, model: str | None = None) -> list[list[float]]:
        effective_model = model if model is not None else self.model
        request_body = OllamaEmbedRequest(model=effective_model, input=inputs)
        char_count = sum(len(s) for s in inputs)
        logger.debug(
            "ollama_embed_request model=%s input_count=%s input_chars=%s",
            effective_model,
            len(inputs),
            char_count,
        )
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{self.base_url}/api/embed",
                    json=request_body.model_dump(mode="json", exclude_none=True),
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error(
                "ollama_embed_http_error model=%s url=%s error=%s",
                effective_model,
                self.base_url,
                exc,
            )
            message = f"Ollama embedding request failed for model {effective_model}"
            raise EmbeddingTransportError(message) from exc

        try:
            body = parse_ollama_json(response.json(), OllamaEmbedResponse)
        except (ValueError, TypeError) as exc:
            logger.error("ollama_embed_invalid_response model=%s error=%s", effective_model, exc)
            message = f"Ollama returned an invalid embedding response for model {effective_model}"
            raise EmbeddingResponseError(message) from exc

        if len(body.embeddings) != len(inputs):
            logger.error(
                "ollama_embed_row_count_mismatch model=%s expected=%s got=%s",
                effective_model,
                len(inputs),
                len(body.embeddings),
            )
            raise EmbeddingResponseError(
                "Ollama returned a different number of embeddings than inputs; "
                "check OLLAMA_EMBED_MODEL and server version."
            )

        result = validate_vectors(
            body.embeddings, provider=self.provider_name, model=effective_model
        )
        self.dimension = len(result[0]) if result else self.dimension
        return result

    def close(self) -> None:
        """Release provider resources (Ollama uses per-call clients)."""
        return
