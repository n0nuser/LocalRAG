from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class OllamaEmbedder:
    base_url: str
    model: str
    timeout_seconds: float = 120.0

    def embed_text(self, text: str) -> list[float]:
        payload = {"model": self.model, "input": text}
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/api/embeddings", json=payload)
            response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        if isinstance(embedding, list):
            return [float(value) for value in embedding]
        raise ValueError("Ollama embedding response missing 'embedding'")

    def embed_texts(self, texts: list[str], batch_size: int) -> list[list[float]]:
        embeddings: list[list[float]] = []
        safe_batch_size = max(1, batch_size)
        for start in range(0, len(texts), safe_batch_size):
            batch = texts[start : start + safe_batch_size]
            for text in batch:
                embeddings.append(self.embed_text(text))
        return embeddings
