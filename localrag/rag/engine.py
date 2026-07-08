from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from localrag.llm.providers.base import BaseLLMProvider
from localrag.rag.prompt import build_prompt
from localrag.rag.retriever import Retriever
from localrag.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class RAGEngine:
    settings: Settings
    retriever: Retriever
    provider: BaseLLMProvider

    def answer(
        self,
        question: str,
        model: str | None = None,
        n_results: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        chunks: list[str] = []
        sources: list[dict[str, object]] = []
        for event in self.stream_answer(
            question=question, model=model, n_results=n_results, metadata_filter=metadata_filter
        ):
            if event["type"] == "token":
                chunks.append(str(event["token"]))
            if event["type"] == "final":
                sources = list(event["sources"])
        return {"answer": "".join(chunks).strip(), "sources": sources}

    def stream_answer(
        self,
        question: str,
        model: str | None = None,
        n_results: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> Generator[dict[str, Any]]:
        logger.info(
            "rag_stream_start question_chars=%s model=%s n_results=%s",
            len(question),
            model,
            n_results,
        )
        contexts = self.retriever.retrieve(
            question=question, n_results=n_results, metadata_filter=metadata_filter
        )
        return self.stream_chat_from_contexts(contexts=contexts, question=question, model=model)

    def stream_chat_from_contexts(
        self,
        *,
        contexts: list[dict[str, Any]],
        question: str,
        model: str | None,
    ) -> Generator[dict[str, Any]]:
        """Stream LLM tokens when contexts were retrieved earlier (HTTP runs retrieve first)."""
        if self._is_low_confidence(contexts):
            logger.info("rag_low_confidence_refusal question_chars=%s", len(question))
            return self._low_confidence_response()
        return self._stream_chat_tokens(contexts=contexts, question=question, model=model)

    def _is_low_confidence(self, contexts: list[dict[str, Any]]) -> bool:
        min_score = self.settings.rag_min_context_score
        if min_score <= 0:
            return False
        if not contexts:
            return True
        top_score = float(contexts[0].get("score", 0.0))
        return top_score < min_score

    @staticmethod
    def _low_confidence_response() -> Generator[dict[str, Any]]:
        yield {
            "type": "token",
            "token": "I don't have enough information in the ingested documents to answer that.",
        }
        yield {"type": "final", "sources": [], "low_confidence": True}

    def _stream_chat_tokens(
        self,
        *,
        contexts: list[dict[str, Any]],
        question: str,
        model: str | None,
    ) -> Generator[dict[str, Any]]:
        logger.debug("rag_contexts count=%s", len(contexts))
        prompt = build_prompt(
            system_prompt=self.settings.rag_system_prompt,
            question=question,
            contexts=contexts,
        )
        for event in self.provider.stream_from_prompt(prompt, model=model):
            if event["type"] == "token":
                yield event
        logger.info("rag_stream_done")
        yield {"type": "final", "sources": self._extract_sources(contexts), "low_confidence": False}

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
            metadata = context.get("metadata") or {}
            sources.append(
                {
                    "source": source,
                    "chunk_index": chunk_index,
                    "heading_path": metadata.get("heading_path") or None,
                    "chunk_type": metadata.get("chunk_type") or None,
                }
            )
        return sources
