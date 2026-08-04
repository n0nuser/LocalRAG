from __future__ import annotations

from localrag.rag.compressor import CompressionBudget, compress_contexts, count_tokens


def _context(text: str, source: str = "a.md", index: int = 0) -> dict[str, object]:
    return {
        "text": text,
        "source": source,
        "chunk_index": index,
        "metadata": {"heading_path": "Guide > Setup"},
    }


def test_compression_is_deterministic_bounded_and_provenance_preserving() -> None:
    result = compress_contexts(
        [_context("noise. Python setup uses uv. More noise.")],
        "How does Python setup use uv?",
        CompressionBudget(
            per_context_tokens=5, total_tokens=5, per_context_chars=40, total_chars=40
        ),
    )

    assert result == compress_contexts(
        [_context("noise. Python setup uses uv. More noise.")],
        "How does Python setup use uv?",
        CompressionBudget(
            per_context_tokens=5, total_tokens=5, per_context_chars=40, total_chars=40
        ),
    )
    context = result.contexts[0]
    assert context["source"] == "a.md"
    assert context["chunk_index"] == 0
    assert context["compression"]["parent_id"] == "Guide > Setup"
    assert result.output_tokens <= 5
    assert result.output_chars <= 40
    assert result.status == "compressed"


def test_empty_and_oversized_indivisible_blocks_return_no_context() -> None:
    budget = CompressionBudget(
        per_context_tokens=2, total_tokens=2, per_context_chars=20, total_chars=20
    )
    assert compress_contexts([], "question", budget).status == "no_context"
    code = _context("```python\nprint('this block is too long')\n```")
    result = compress_contexts([code], "print", budget)
    assert result.status == "no_context"
    assert result.contexts == []


def test_table_is_kept_as_one_safe_unit() -> None:
    table = _context("| Name | Value |\n| --- | --- |\n| alpha | 1 |")
    result = compress_contexts(
        [table],
        "alpha",
        CompressionBudget(
            per_context_tokens=20, total_tokens=20, per_context_chars=100, total_chars=100
        ),
    )
    assert result.contexts[0]["text"] == table["text"]


def test_duplicate_contexts_are_retained_as_distinct_citations_but_overlap_is_ordered() -> None:
    contexts = [
        _context("first answer. second answer.", "a.md", 0),
        _context("first answer.", "a.md", 1),
    ]
    result = compress_contexts(
        contexts,
        "answer",
        CompressionBudget(
            max_contexts=2,
            per_context_tokens=10,
            total_tokens=10,
            per_context_chars=100,
            total_chars=100,
        ),
    )
    assert [item["chunk_index"] for item in result.contexts] == [0, 1]
    assert result.contexts[0]["text"].startswith("first answer.")
    assert count_tokens("你好 世界") == 2


def test_scorer_failure_uses_bounded_fallback() -> None:
    result = compress_contexts(
        [_context("one. two.")],
        "question",
        CompressionBudget(
            per_context_tokens=1, total_tokens=1, per_context_chars=10, total_chars=10
        ),
        scorer=lambda _question, _text: (_ for _ in ()).throw(TimeoutError("slow")),
    )
    assert result.status == "fallback"
    assert result.output_tokens <= 1
    assert result.output_chars <= 10
    assert result.contexts[0]["compression"]["status"] == "fallback"
