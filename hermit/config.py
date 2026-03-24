from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_llm_model: str = "llama3.2"

    chroma_persist_path: str = "./data/chroma"
    chroma_collection_name: str = "hermit"

    chunk_chars: int = 512
    chunk_overlap_chars: int = 50
    embedding_batch_size: int = 32

    ingest_recursive: bool = True
    ingest_roots: list[str] = []

    rag_top_k: int = 5
    rag_system_prompt: str = (
        "You are a helpful assistant. Answer only based on the provided context."
    )

    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def is_path_allowed(candidate: Path, roots: list[str]) -> bool:
    if not roots:
        return True

    resolved_candidate = candidate.resolve()
    for root in roots:
        resolved_root = Path(root).resolve()
        if resolved_candidate == resolved_root or resolved_root in resolved_candidate.parents:
            return True
    return False
