from __future__ import annotations

import json
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import httpx

from hermit.config import Settings
from hermit.rag.prompt import build_prompt
from hermit.rag.retriever import Retriever


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
        contexts = self.retriever.retrieve(question=question, n_results=n_results)
        prompt = build_prompt(
            system_prompt=self.settings.rag_system_prompt,
            question=question,
            contexts=contexts,
        )
        runtime_model = model or self.settings.ollama_llm_model
        payload = {
            "model": runtime_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        with (
            httpx.Client(timeout=self.timeout_seconds) as client,
            client.stream(
                "POST", f"{self.settings.ollama_base_url}/api/chat", json=payload
            ) as resp,
        ):
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                parsed = self._parse_ollama_line(line)
                token = parsed.get("message", {}).get("content", "")
                if token:
                    yield {"type": "token", "token": token}
                if parsed.get("done"):
                    break

        yield {"type": "final", "sources": self._extract_sources(contexts)}

    @staticmethod
    def _parse_ollama_line(line: str) -> dict[str, Any]:
        return json.loads(line)

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
