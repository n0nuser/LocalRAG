from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from hermit.ollama.schemas import (
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

    def embed_text(self, text: str) -> list[float]:
        rows = self._embed_inputs([text])
        return rows[0]

    def embed_texts(self, texts: list[str], batch_size: int) -> list[list[float]]:
        if not texts:
            return []
        safe_batch_size = max(1, batch_size)
        out: list[list[float]] = []
        logger.debug(
            "ollama_embed_batch total_texts=%s batch_size=%s",
            len(texts),
            safe_batch_size,
        )
        for start in range(0, len(texts), safe_batch_size):
            batch = texts[start : start + safe_batch_size]
            out.extend(self._embed_inputs(batch))
        return out

    def _embed_inputs(self, inputs: list[str]) -> list[list[float]]:
        request_body = OllamaEmbedRequest(model=self.model, input=inputs)
        char_count = sum(len(s) for s in inputs)
        logger.debug(
            "ollama_embed_request model=%s input_count=%s input_chars=%s",
            self.model,
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
                self.model,
                self.base_url,
                exc,
            )
            raise

        try:
            body = parse_ollama_json(response.json(), OllamaEmbedResponse)
        except ValueError as exc:
            logger.error("ollama_embed_invalid_response model=%s error=%s", self.model, exc)
            raise

        if len(body.embeddings) != len(inputs):
            logger.error(
                "ollama_embed_row_count_mismatch model=%s expected=%s got=%s",
                self.model,
                len(inputs),
                len(body.embeddings),
            )
            raise ValueError(
                "Ollama returned a different number of embeddings than inputs; "
                "check OLLAMA_EMBED_MODEL and server version."
            )

        result: list[list[float]] = []
        for index, row in enumerate(body.embeddings):
            vector = [float(value) for value in row]
            if not vector:
                logger.error(
                    "ollama_embed_empty_vector model=%s embedding_index=%s",
                    self.model,
                    index,
                )
                raise ValueError(
                    "Ollama returned an empty embedding vector; "
                    "check OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL, and that the model is pulled."
                )
            result.append(vector)
        return result
