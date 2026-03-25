from __future__ import annotations

import httpx
import pytest
import respx

from hermit.ingestion.embedder import OllamaEmbedder


@respx.mock
def test_embed_text_uses_api_embed_and_input() -> None:
    route = respx.post("http://ollama:11434/api/embed").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "nomic-embed-text",
                "embeddings": [[0.1, 0.2, 0.3]],
            },
        )
    )
    embedder = OllamaEmbedder(base_url="http://ollama:11434", model="nomic-embed-text")
    result = embedder.embed_text("hello")

    assert result == [0.1, 0.2, 0.3]
    sent = route.calls[0].request.content.decode()
    assert '"input"' in sent
    assert "/api/embed" in str(route.calls[0].request.url)


@respx.mock
def test_embed_text_rejects_empty_vector() -> None:
    respx.post("http://ollama:11434/api/embed").mock(
        return_value=httpx.Response(
            200,
            json={"model": "nomic-embed-text", "embeddings": [[]]},
        ),
    )
    embedder = OllamaEmbedder(base_url="http://ollama:11434", model="nomic-embed-text")
    with pytest.raises(ValueError, match="empty embedding"):
        embedder.embed_text("x")


@respx.mock
def test_embed_texts_batches_match_row_count() -> None:
    respx.post("http://ollama:11434/api/embed").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "nomic-embed-text",
                "embeddings": [[1.0], [2.0]],
            },
        ),
    )
    embedder = OllamaEmbedder(base_url="http://ollama:11434", model="nomic-embed-text")
    out = embedder.embed_texts(["a", "b"], batch_size=2)
    assert out == [[1.0], [2.0]]
