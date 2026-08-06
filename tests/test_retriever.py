from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus

import httpx
import pytest
import respx

from localrag.ingestion.embedder import OllamaEmbedder
from localrag.rag.exceptions import RetrievalError
from localrag.rag.query_rewrite import QueryExpansionResult
from localrag.rag.retriever import Retriever
from localrag.settings import Settings


@dataclass
class StubEmbedder:
    def embed_text(self, text: str, *, model: str | None = None) -> list[float]:
        return [1.0, 2.0, 3.0]


@dataclass
class StubStore:
    def query(
        self, embedding: list[float], top_k: int, where: dict[str, object] | None = None
    ) -> dict[str, object]:
        _ = (embedding, top_k, where)
        return {
            "documents": [["chunk-a"]],
            "metadatas": [[{"source": "foo.md", "chunk_index": 0}]],
            "distances": [[0.12]],
        }


def test_retriever_returns_contexts() -> None:
    settings = Settings()
    retriever = Retriever(
        settings=settings,
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        vector_store=StubStore(),  # type: ignore[arg-type]
    )

    contexts = retriever.retrieve("hello")

    assert contexts == [
        {
            "text": "chunk-a",
            "source": "foo.md",
            "chunk_index": 0,
            "score": pytest.approx(0.8928571428571428),
            "distance": 0.12,
            "ingested_at": None,
            "metadata": {"source": "foo.md", "chunk_index": 0},
            "freshness_factor": 1.0,
        }
    ]


def test_retriever_for_collection_creates_request_scoped_store() -> None:
    @dataclass
    class CollectionStore(StubStore):
        selected: list[str]

        def for_collection(self, name: str) -> CollectionStore:
            self.selected.append(name)
            return self

    settings = Settings(chroma_collection_name="default")
    store = CollectionStore(selected=[])
    retriever = Retriever(
        settings=settings,
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        vector_store=store,  # type: ignore[arg-type]
    )

    selected = retriever.for_collection("experiments")

    assert store.selected == ["experiments"]
    assert selected.settings.chroma_collection_name == "experiments"
    assert selected.embedder is retriever.embedder


@respx.mock
def test_retriever_raises_retrieval_failure_when_ollama_embed_fails() -> None:
    respx.post("http://ollama:11434/api/embed").mock(return_value=httpx.Response(503))
    embedder = OllamaEmbedder(base_url="http://ollama:11434", model="nomic-embed-text")
    retriever = Retriever(
        settings=Settings(),
        embedder=embedder,
        vector_store=StubStore(),  # type: ignore[arg-type]
    )

    with pytest.raises(RetrievalError) as excinfo:
        retriever.retrieve("q")

    assert excinfo.value.status_code == HTTPStatus.BAD_GATEWAY


def test_retriever_threads_metadata_filter_to_vector_store_where() -> None:
    captured: dict[str, object] = {}

    @dataclass
    class CapturingStore:
        @staticmethod
        def query(
            embedding: list[float], top_k: int, where: dict[str, object] | None = None
        ) -> dict[str, object]:
            _ = (embedding, top_k)
            captured["where"] = where
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    retriever = Retriever(
        settings=Settings(),
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        vector_store=CapturingStore(),  # type: ignore[arg-type]
    )

    retriever.retrieve("q", metadata_filter={"source": "a.md"})

    assert captured["where"] == {"source": "a.md"}


def test_retriever_raises_retrieval_failure_when_vector_query_fails() -> None:
    @dataclass
    class ExplodingStore:
        def query(self, embedding: list[float], top_k: int) -> dict[str, object]:
            raise RuntimeError("dimension mismatch")

    retriever = Retriever(
        settings=Settings(),
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        vector_store=ExplodingStore(),  # type: ignore[arg-type]
    )

    with pytest.raises(RetrievalError) as excinfo:
        retriever.retrieve("q")

    assert excinfo.value.status_code == HTTPStatus.SERVICE_UNAVAILABLE


def test_retriever_expands_top_hits_to_full_heading_section() -> None:
    @dataclass
    class ExpandableStore:
        @staticmethod
        def query(
            embedding: list[float], top_k: int, where: dict[str, object] | None = None
        ) -> dict[str, object]:
            _ = (embedding, top_k, where)
            return {
                "documents": [["Section intro sentence."]],
                "metadatas": [[{"source": "guide.md", "chunk_index": 0, "heading_path": "Setup"}]],
                "distances": [[0.05]],
            }

        @staticmethod
        def get_chunks_by_headings(
            headings: list[tuple[str, str]], metadata_filter: dict[str, object] | None = None
        ) -> dict[tuple[str, str], list[tuple[int, str]]]:
            assert headings == [("guide.md", "Setup")]
            assert metadata_filter is None
            return {
                ("guide.md", "Setup"): [
                    (0, "Section intro sentence."),
                    (1, "Second sentence with the install command."),
                ]
            }

    retriever = Retriever(
        settings=Settings(),
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        vector_store=ExpandableStore(),  # type: ignore[arg-type]
    )

    contexts = retriever.retrieve("how do I install this")

    assert contexts[0]["expanded_text"] == (
        "Section intro sentence.\n\nSecond sentence with the install command."
    )
    assert contexts[0]["text"] == "Section intro sentence."


def test_retriever_expands_unique_parents_once_and_preserves_missing_hits() -> None:
    @dataclass
    class CountingStore:
        bulk_calls: list[tuple[list[tuple[str, str]], dict[str, object] | None]]

        def query(
            self, embedding: list[float], top_k: int, where: dict[str, object] | None = None
        ) -> dict[str, object]:
            _ = (embedding, top_k, where)
            return {
                "documents": [["first", "second", "orphan"]],
                "metadatas": [
                    [
                        {"source": "guide.md", "chunk_index": 0, "heading_path": "Setup"},
                        {"source": "guide.md", "chunk_index": 1, "heading_path": "Setup"},
                        {"source": "guide.md", "chunk_index": 2, "heading_path": "Missing"},
                    ]
                ],
                "distances": [[0.1, 0.2, 0.3]],
            }

        def get_chunks_by_headings(
            self, headings: list[tuple[str, str]], metadata_filter: dict[str, object] | None = None
        ) -> dict[tuple[str, str], list[tuple[int, str]]]:
            self.bulk_calls.append((headings, metadata_filter))
            return {
                ("guide.md", "Setup"): [(0, "first"), (1, "second")],
            }

    store = CountingStore(bulk_calls=[])
    retriever = Retriever(
        settings=Settings(retrieval_mode="vector"),
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        vector_store=store,  # type: ignore[arg-type]
    )

    contexts = retriever.retrieve("q", n_results=3)

    assert len(store.bulk_calls) == 1
    assert set(store.bulk_calls[0][0]) == {
        ("guide.md", "Setup"),
        ("guide.md", "Missing"),
    }
    assert [(context["text"], context["chunk_index"]) for context in contexts] == [
        ("first", 0),
        ("second", 1),
        ("orphan", 2),
    ]
    assert contexts[0]["expanded_text"] == "first\n\nsecond"
    assert "expanded_text" not in contexts[2]


def test_retriever_passes_filter_to_parent_expansion_and_filters_siblings() -> None:
    @dataclass
    class FilteredStore:
        filters: list[dict[str, object] | None]

        def query(
            self, embedding: list[float], top_k: int, where: dict[str, object] | None = None
        ) -> dict[str, object]:
            _ = (embedding, top_k)
            assert where == {"tenant_id": "team-a"}
            return {
                "documents": [["team-a match"]],
                "metadatas": [
                    [
                        {
                            "source": "guide.md",
                            "chunk_index": 1,
                            "heading_path": "Setup",
                            "tenant_id": "team-a",
                        }
                    ]
                ],
                "distances": [[0.1]],
            }

        def get_chunks_by_headings(
            self, headings: list[tuple[str, str]], metadata_filter: dict[str, object] | None = None
        ) -> dict[tuple[str, str], list[tuple[int, str]]]:
            _ = headings
            self.filters.append(metadata_filter)
            assert metadata_filter == {"tenant_id": "team-a"}
            return {("guide.md", "Setup"): [(1, "team-a match"), (2, "team-a sibling")]}

    store = FilteredStore(filters=[])
    retriever = Retriever(
        settings=Settings(retrieval_mode="vector"),
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        vector_store=store,  # type: ignore[arg-type]
    )

    contexts = retriever.retrieve("q", metadata_filter={"tenant_id": "team-a"})

    assert store.filters == [{"tenant_id": "team-a"}]
    assert contexts[0]["expanded_text"] == "team-a match\n\nteam-a sibling"


def test_retriever_applies_reranker_over_widened_candidate_pool() -> None:
    @dataclass
    class TwoDocStore:
        @staticmethod
        def query(
            embedding: list[float], top_k: int, where: dict[str, object] | None = None
        ) -> dict[str, object]:
            _ = (embedding, top_k, where)
            return {
                "documents": [["nearby vector text", "exact token text"]],
                "metadatas": [
                    [
                        {"source": "nearby.md", "chunk_index": 0},
                        {"source": "exact.md", "chunk_index": 1},
                    ]
                ],
                "distances": [[0.01, 0.4]],
            }

    @dataclass
    class FakeReranker:
        calls: list[tuple[str, int]]

        def rerank(
            self, question: str, contexts: list[dict[str, object]], top_k: int
        ) -> list[dict[str, object]]:
            self.calls.append((question, top_k))
            return list(reversed(contexts))[:top_k]

    reranker = FakeReranker(calls=[])
    retriever = Retriever(
        settings=Settings(retrieval_mode="vector", rerank_fetch_k=2),
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        vector_store=TwoDocStore(),  # type: ignore[arg-type]
        reranker=reranker,  # type: ignore[arg-type]
    )

    contexts = retriever.retrieve("q", n_results=1)

    assert reranker.calls == [("q", 1)]
    assert contexts[0]["source"] == "exact.md"


def test_retriever_skips_expansion_when_heading_path_empty() -> None:
    @dataclass
    class FlatStore:
        @staticmethod
        def query(
            embedding: list[float], top_k: int, where: dict[str, object] | None = None
        ) -> dict[str, object]:
            _ = (embedding, top_k, where)
            return {
                "documents": [["plain text chunk"]],
                "metadatas": [[{"source": "notes.txt", "chunk_index": 0, "heading_path": ""}]],
                "distances": [[0.05]],
            }

    retriever = Retriever(
        settings=Settings(),
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        vector_store=FlatStore(),  # type: ignore[arg-type]
    )

    contexts = retriever.retrieve("q")

    assert "expanded_text" not in contexts[0]


def test_retriever_uses_rewritten_query_for_embedding_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "localrag.rag.retriever.rewrite_query",
        lambda question, _settings: "rewritten " + question,
    )
    seen_questions: list[str] = []

    @dataclass
    class RecordingEmbedder:
        def embed_text(self, text: str, *, model: str | None = None) -> list[float]:
            seen_questions.append(text)
            return [0.1, 0.2, 0.3]

    retriever = Retriever(
        settings=Settings(query_rewrite_enabled=True),
        embedder=RecordingEmbedder(),  # type: ignore[arg-type]
        vector_store=StubStore(),  # type: ignore[arg-type]
    )

    retriever.retrieve("original question")

    assert seen_questions == ["rewritten original question"]


def test_retriever_expands_with_bounded_variants_and_post_fusion_reranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "localrag.rag.retriever.expand_query",
        lambda question, search, _settings, rewrite=None: QueryExpansionResult(
            question, rewrite, (question, search, "alternative"), (), "expanded"
        ),
    )
    seen_queries: list[str] = []

    @dataclass
    class RecordingEmbedder:
        def embed_text(self, text: str, *, model: str | None = None) -> list[float]:
            _ = model
            seen_queries.append(text)
            return [1.0, 2.0, 3.0]

    @dataclass
    class OneResultStore:
        calls: int = 0

        def query(
            self, embedding: list[float], top_k: int, where: dict[str, object] | None = None
        ) -> dict[str, object]:
            _ = (embedding, top_k)
            assert where == {"tenant_id": "team-a"}
            self.calls += 1
            if self.calls > 1:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            return {
                "documents": [["matching chunk"]],
                "metadatas": [[{"source": "a.md", "chunk_index": 0, "tenant_id": "team-a"}]],
                "distances": [[0.1]],
            }

    @dataclass
    class RecordingReranker:
        calls: list[tuple[str, int]]

        def rerank(
            self, question: str, contexts: list[dict[str, object]], top_k: int
        ) -> list[dict[str, object]]:
            self.calls.append((question, len(contexts)))
            return contexts[:top_k]

    reranker = RecordingReranker([])
    retriever = Retriever(
        settings=Settings(
            query_expansion_enabled=True,
            retrieval_mode="vector",
            query_expansion_candidate_budget=2,
        ),
        embedder=RecordingEmbedder(),  # type: ignore[arg-type]
        vector_store=OneResultStore(),  # type: ignore[arg-type]
        reranker=reranker,  # type: ignore[arg-type]
    )

    contexts = retriever.retrieve("original ERR_CODE", metadata_filter={"tenant_id": "team-a"})

    assert seen_queries == ["original ERR_CODE", "original ERR_CODE", "alternative"]
    assert reranker.calls == [("original ERR_CODE", 1)]
    assert contexts[0]["source"] == "a.md"
