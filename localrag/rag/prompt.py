from __future__ import annotations

from typing import Any

# Constraining the model to the context is not enough on its own: a passage about
# habitual exposure measured over years and one about a single occurrence are both
# just "context", and the model will flatten them together unless told not to. The
# second and third sentences are what stop a chronic finding being restated as an
# acute one.
#
# Length is a deliberate constraint. This ships on every query and small local
# models follow long instructions unevenly, so the scope rule is stated as two
# imperatives rather than a policy paragraph.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer only based on the provided context. "
    "Preserve the scope each source states: if a passage describes repeated, "
    "habitual, or long-term exposure, do not present it as the result of a single "
    "occurrence. When the question asks about one instance and the context only "
    "supports long-term findings, say so instead of answering as if it did."
)

# A heading path is unbounded in depth, so a deeply nested one could crowd out the
# chunk body it is meant to qualify. Truncating keeps the leading levels, which carry
# the coarse scope (chapter, section) that disambiguates the passage.
#
# The bound is set against the compression reservation rather than chosen for looks:
# ``reserved_prompt_tokens`` (512 by default) covers everything outside the chunk
# bodies, and the worst case is ``max_contexts`` headings of single-character words,
# i.e. one whitespace token per two characters. 120 chars keeps five such headings
# plus the block scaffolding inside that reservation.
MAX_SECTION_CHARS = 120


def build_prompt(system_prompt: str, question: str, contexts: list[dict[str, object]]) -> str:
    context_blocks: list[str] = []
    for index, context in enumerate(contexts, start=1):
        source = context.get("source", "unknown")
        chunk_index = context.get("chunk_index", -1)
        text = context.get("expanded_text") or context.get("text", "")
        section = _section(context.get("metadata"))
        header = f"[{index}] source={source} chunk={chunk_index}"
        if section:
            header = f"{header} section={section}"
        context_blocks.append(f"{header}\n{text}")

    joined_context = "\n\n".join(context_blocks) if context_blocks else "No context found."
    return f"{system_prompt}\n\nContext:\n{joined_context}\n\nQuestion:\n{question}\n\nAnswer:"


def _section(metadata: Any) -> str:
    """Return the bounded heading path, or an empty string when there is none.

    Only structural (markdown) chunks carry a heading path; text and code blocks
    store an empty one, and the ollama provider builds contexts with no metadata
    at all, so absence is normal rather than an error.
    """
    if not isinstance(metadata, dict):
        return ""
    heading_path = str(metadata.get("heading_path") or "").strip()
    if len(heading_path) > MAX_SECTION_CHARS:
        return heading_path[: MAX_SECTION_CHARS - 1] + "…"
    return heading_path
