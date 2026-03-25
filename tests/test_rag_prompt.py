from __future__ import annotations

from localrag.rag.prompt import build_prompt


def test_build_prompt_without_context() -> None:
    out = build_prompt(system_prompt="SYS", question="Q", contexts=[])
    assert "SYS" in out
    assert "Context:\nNo context found." in out
    assert "Question:\nQ" in out
    assert out.endswith("\n\nAnswer:")


def test_build_prompt_includes_context_blocks() -> None:
    contexts = [
        {"source": "foo.md", "chunk_index": 2, "text": "hello"},
        {"source": "bar.md", "chunk_index": 0, "text": "world"},
    ]
    out = build_prompt(system_prompt="SYS", question="Q", contexts=contexts)

    assert "[1] source=foo.md chunk=2\nhello" in out
    assert "[2] source=bar.md chunk=0\nworld" in out
    assert "Question:\nQ" in out
