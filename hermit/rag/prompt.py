from __future__ import annotations


def build_prompt(system_prompt: str, question: str, contexts: list[dict[str, object]]) -> str:
    context_blocks: list[str] = []
    for index, context in enumerate(contexts, start=1):
        source = context.get("source", "unknown")
        chunk_index = context.get("chunk_index", -1)
        text = context.get("text", "")
        context_blocks.append(f"[{index}] source={source} chunk={chunk_index}\n{text}")

    joined_context = "\n\n".join(context_blocks) if context_blocks else "No context found."
    return f"{system_prompt}\n\nContext:\n{joined_context}\n\nQuestion:\n{question}\n\nAnswer:"
