from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

CHUNK_CONTRACT_VERSION = "1"


@dataclass
class Chunk:
    """The common indexing result shared by every chunking strategy.

    Offsets are currently absent (``None``): strategies strip whitespace and
    structural chunking repacks blocks, so source offsets would be misleading.
    ``metadata`` carries strategy-specific fields without changing this shape.
    """

    text: str
    heading_path: str = ""
    chunk_type: str = "text_block"
    chunk_index: int = 0
    source: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    parent_id: str | None = None
    chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def stable_chunk_id(source: str, strategy: str, index: int, text: str) -> str:
    """Return a deterministic ID for one logical chunk in a source."""
    identity = "\0".join((CHUNK_CONTRACT_VERSION, source, strategy, str(index), text))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()
