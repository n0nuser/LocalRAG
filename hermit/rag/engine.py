from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import httpx

from hermit.ollama.schemas import (
    OllamaChatMessage,
    OllamaChatRequest,
    OllamaChatStreamChunk,
    parse_ollama_json_line,
)
from hermit.rag.prompt import build_prompt
from hermit.rag.retriever import Retriever
from hermit.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class RAGEngine:
    settings: Settings
    retriever: Retriever
    timeout_seconds: float = 180.0

    def answer(
        self, question: str, model: str | None = None, n_results: int | None = None
    ) -> dict[str, object]:
        chunks: list[str] = []
        sources: list[dict[str, object]] = []
        for event in self.stream_answer(question=question, model=model, n_results=n_results):
            if event["type"] == "token":
                chunks.append(str(event["token"]))
            if event["type"] == "final":
                sources = list(event["sources"])
        return {"answer": "".join(chunks).strip(), "sources": sources}

    def stream_answer(
        self, question: str, model: str | None = None, n_results: int | None = None
    ) -> Generator[dict[str, Any]]:
        logger.info(
            "rag_stream_start question_chars=%s model=%s n_results=%s",
            len(question),
            model,
            n_results,
        )
        contexts = self.retriever.retrieve(question=question, n_results=n_results)
        logger.debug("rag_contexts count=%s", len(contexts))
        prompt = build_prompt(
            system_prompt=self.settings.rag_system_prompt,
            question=question,
            contexts=contexts,
        )
        runtime_model = model or self.settings.ollama_llm_model
        chat_request = OllamaChatRequest(
            model=runtime_model,
            messages=[OllamaChatMessage(role="user", content=prompt)],
            stream=True,
        )

        try:
            with (
                httpx.Client(timeout=self.timeout_seconds) as client,
                client.stream(
                    "POST",
                    f"{self.settings.ollama_base_url}/api/chat",
                    json=chat_request.model_dump(mode="json", exclude_none=True),
                ) as resp,
            ):
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = parse_ollama_json_line(line, OllamaChatStreamChunk)
                    except ValueError:
                        logger.warning("rag_ollama_bad_chunk line_chars=%s", len(line))
                        continue
                    msg = chunk.message
                    token = msg.content if msg and msg.content else ""
                    if token:
                        yield {"type": "token", "token": token}
                    if chunk.done:
                        break
        except httpx.HTTPError as exc:
            logger.error(
                "rag_ollama_chat_http_error url=%s model=%s error=%s",
                self.settings.ollama_base_url,
                runtime_model,
                exc,
            )
            raise

        logger.info("rag_stream_done model=%s", runtime_model)
        yield {"type": "final", "sources": self._extract_sources(contexts)}

    @staticmethod
    def _extract_sources(contexts: list[dict[str, Any]]) -> list[dict[str, object]]:
        seen: set[tuple[str, int]] = set()
        sources: list[dict[str, object]] = []
        for context in contexts:
            source = str(context.get("source", "unknown"))
            chunk_index = int(context.get("chunk_index", -1))
            key = (source, chunk_index)
            if key in seen:
                continue
            seen.add(key)
            sources.append({"source": source, "chunk_index": chunk_index})
        return sources
