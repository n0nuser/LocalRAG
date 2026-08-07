from __future__ import annotations

from localrag.rag.compressor import count_tokens
from localrag.rag.prompt import MAX_SECTION_CHARS, build_prompt
from localrag.settings_groups import ContextCompressionSettings


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


def test_build_prompt_includes_heading_path_as_section() -> None:
    contexts = [
        {
            "source": "book.md",
            "chunk_index": 473,
            "text": "hello",
            "metadata": {"heading_path": "Cancer > SLEEP LOSS AND THE CARDIOVASCULAR SYSTEM"},
        }
    ]
    out = build_prompt(system_prompt="SYS", question="Q", contexts=contexts)

    assert (
        "[1] source=book.md chunk=473 "
        "section=Cancer > SLEEP LOSS AND THE CARDIOVASCULAR SYSTEM\nhello"
    ) in out


def test_build_prompt_omits_section_when_heading_path_is_absent() -> None:
    contexts = [
        {"source": "a.md", "chunk_index": 0, "text": "no metadata key at all"},
        {"source": "b.md", "chunk_index": 1, "text": "empty heading", "metadata": {}},
        {
            "source": "c.md",
            "chunk_index": 2,
            "text": "blank heading",
            "metadata": {"heading_path": ""},
        },
        {
            "source": "d.md",
            "chunk_index": 3,
            "text": "whitespace heading",
            "metadata": {"heading_path": "   "},
        },
    ]
    out = build_prompt(system_prompt="SYS", question="Q", contexts=contexts)

    assert "section=" not in out
    assert "[1] source=a.md chunk=0\nno metadata key at all" in out
    assert "[4] source=d.md chunk=3\nwhitespace heading" in out


def test_build_prompt_tolerates_non_mapping_metadata() -> None:
    contexts = [{"source": "a.md", "chunk_index": 0, "text": "body", "metadata": "not-a-dict"}]
    out = build_prompt(system_prompt="SYS", question="Q", contexts=contexts)

    assert "[1] source=a.md chunk=0\nbody" in out


def test_build_prompt_truncates_a_long_heading_path() -> None:
    heading_path = "A" * (MAX_SECTION_CHARS + 50)
    contexts = [
        {
            "source": "a.md",
            "chunk_index": 0,
            "text": "body",
            "metadata": {"heading_path": heading_path},
        }
    ]
    out = build_prompt(system_prompt="SYS", question="Q", contexts=contexts)

    header = out.split("Context:\n", 1)[1].split("\n", 1)[0]
    section = header.split("section=", 1)[1]
    assert len(section) == MAX_SECTION_CHARS
    assert section.endswith("…")
    assert heading_path not in out


def test_section_headers_stay_inside_the_reserved_prompt_budget() -> None:
    """The heading bound is only safe if worst-case headers fit the reservation.

    Compression budgets only ever measure chunk body text, so everything the prompt
    adds around those bodies has to fit ``reserved_prompt_tokens``. The worst case is
    a full set of contexts whose heading paths are single-character words, which cost
    one whitespace token per two characters.
    """
    budget = ContextCompressionSettings()
    worst_heading = " ".join("x" for _ in range(MAX_SECTION_CHARS // 2))
    contexts: list[dict[str, object]] = [
        {
            "source": "a" * 60 + ".md",
            "chunk_index": 999999,
            "text": "",
            "metadata": {"heading_path": worst_heading},
        }
        for _ in range(budget.max_contexts)
    ]
    overhead = build_prompt(system_prompt="", question="", contexts=contexts)

    assert count_tokens(overhead) <= budget.reserved_prompt_tokens


def test_build_prompt_prefers_expanded_text_over_matched_text() -> None:
    contexts = [
        {
            "source": "guide.md",
            "chunk_index": 0,
            "text": "matched sentence only",
            "expanded_text": "full section text including matched sentence",
        }
    ]
    out = build_prompt(system_prompt="SYS", question="Q", contexts=contexts)

    assert "full section text including matched sentence" in out
    assert "matched sentence only" not in out
