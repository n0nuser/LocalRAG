from __future__ import annotations

import pytest

from hermit.ollama.schemas import (
    OllamaChatStreamChunk,
    OllamaEmbedResponse,
    OllamaTagsResponse,
    parse_ollama_json,
    parse_ollama_json_line,
)


def test_parse_tags_round_trip() -> None:
    raw = {
        "models": [
            {
                "name": "nomic-embed-text:latest",
                "model": "nomic-embed-text:latest",
                "size": 123,
                "digest": "abc",
                "details": {
                    "family": "bert",
                    "families": ["bert"],
                    "parameter_size": "137M",
                    "quantization_level": "F16",
                },
            }
        ]
    }
    parsed = parse_ollama_json(raw, OllamaTagsResponse)
    assert len(parsed.models) == 1
    assert parsed.models[0].name == "nomic-embed-text:latest"


def test_parse_embed_response_requires_embeddings_key() -> None:
    with pytest.raises(ValueError, match="OllamaEmbedResponse"):
        parse_ollama_json({"model": "x"}, OllamaEmbedResponse)


def test_parse_chat_stream_line() -> None:
    line = '{"model":"m","message":{"role":"assistant","content":"Hi"},"done":false}'
    chunk = parse_ollama_json_line(line, OllamaChatStreamChunk)
    assert chunk.message is not None
    assert chunk.message.content == "Hi"
    assert chunk.done is False
