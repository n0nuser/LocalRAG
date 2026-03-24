from __future__ import annotations


def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    cleaned_text = text.strip()
    if not cleaned_text:
        return []
    if chunk_chars <= 0:
        return [cleaned_text]

    step = max(1, chunk_chars - max(0, overlap_chars))
    chunks: list[str] = []
    start = 0

    while start < len(cleaned_text):
        chunk = cleaned_text[start : start + chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks
