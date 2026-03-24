# hermit

Offline-first RAG system. Your documents, your models, your machine.

## What It Is

Hermit ingests your local documents, stores embeddings in a local ChromaDB database,
and answers questions using Ollama models. No cloud services required.

## Quick Start (uv + local Ollama)

1. Install dependencies:

```bash
uv sync
```

1. Start Ollama:

```bash
ollama serve
```

1. Pull models (or let `hermit setup` do it):

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

1. Ingest docs and ask a question:

```bash
uv run hermit ingest ./docs
uv run hermit query "What are the key topics in these documents?"
```

## API

Run the API:

```bash
uv run uvicorn hermit.api.main:app --reload
```

Then open:

- `http://127.0.0.1:8000/docs` for interactive docs
- `GET /health`
- `POST /ingest`
- `POST /ingest/directory`
- `POST /query` (SSE streaming)

## Configuration

Copy `.env.example` to `.env` and tweak values:

```bash
cp .env.example .env
```

Main keys:

- `OLLAMA_BASE_URL`
- `OLLAMA_EMBED_MODEL`
- `OLLAMA_LLM_MODEL`
- `CHROMA_PERSIST_PATH`
- `CHROMA_COLLECTION_NAME`
- `CHUNK_CHARS`
- `CHUNK_OVERLAP_CHARS`
- `INGEST_RECURSIVE`
- `RAG_TOP_K`

## Docker

```bash
docker compose up --build
```

After startup, pull models in the Ollama container:

```bash
docker exec -it <ollama_container_name> ollama pull nomic-embed-text
docker exec -it <ollama_container_name> ollama pull llama3.2
```
