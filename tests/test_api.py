from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from hermit.api.dependencies import get_engine
from hermit.api.main import app


@dataclass
class StubEngine:
    def stream_answer(
        self, question: str, model: str | None = None, n_results: int | None = None
    ) -> object:
        _ = (question, model, n_results)
        yield {"type": "token", "token": "hello "}
        yield {"type": "token", "token": "world"}
        yield {"type": "final", "sources": [{"source": "doc.md", "chunk_index": 1}]}


def test_query_streams_events() -> None:
    app.dependency_overrides[get_engine] = lambda: StubEngine()
    client = TestClient(app)

    response = client.post("/query", json={"question": "Hi"})

    assert response.status_code == 200
    assert "event: token" in response.text
    assert "hello" in response.text
    assert "event: final" in response.text

    app.dependency_overrides.clear()
