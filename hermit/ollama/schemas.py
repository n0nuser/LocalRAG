"""Pydantic shapes for Ollama HTTP calls Hermit actually makes.

Only request fields we send and response fields we read are declared; everything
else from the wire is ignored via ``extra="ignore"``.

See: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, ValidationError


def parse_ollama_json[TModel: BaseModel](data: object, model: type[TModel]) -> TModel:
    """Validate JSON-like data; wrap :class:`ValidationError` with context."""
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        message = f"Ollama response does not match {model.__name__}: {exc}"
        raise ValueError(message) from exc


def parse_ollama_json_line[TModel: BaseModel](line: str, model: type[TModel]) -> TModel:
    """Parse one JSON line from an NDJSON stream."""
    try:
        return model.model_validate_json(line)
    except ValidationError as exc:
        message = f"Ollama stream line does not match {model.__name__}: {exc}"
        raise ValueError(message) from exc


# --- POST /api/embed ---


class OllamaEmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    input: str | list[str]


class OllamaEmbedResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    embeddings: list[list[float]]


# --- POST /api/chat ---


class OllamaChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[OllamaChatMessage]
    stream: bool = True


class OllamaChatStreamMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str | None = None


class OllamaChatStreamChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: OllamaChatStreamMessage | None = None
    done: bool | None = False


# --- GET /api/tags ---


class OllamaTagsModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str


class OllamaTagsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    models: list[OllamaTagsModel]


# --- POST /api/pull ---


class OllamaPullRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    stream: bool = True


class OllamaPullStreamChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str | None = None
