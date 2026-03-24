# Installing Ollama

Hermit talks to **Ollama** on your machine for local embeddings and chat. Ollama is a separate install—not a Python package.

The **canonical instructions** are on the official site:

- **Home & docs:** [ollama.com](https://ollama.com/)
- **Download / install:** [ollama.com/download](https://ollama.com/download)

Follow the steps there for your OS (Windows, macOS, or Linux). The site covers installers, PATH, and optional GPU notes.

## After installation

1. **Check the CLI** (new terminal after install):

   ```bash
   ollama --version
   ```

2. **Run the server** (Hermit expects it reachable, default `http://127.0.0.1:11434`):

   ```bash
   ollama serve
   ```

   On many setups the Ollama app starts this for you in the background; if `hermit` or the API cannot reach Ollama, run `ollama serve` explicitly.

3. **Pull models** Hermit uses by default (names match [`.env.example`](../.env.example)):

   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

   You can change models via `OLLAMA_EMBED_MODEL` and `OLLAMA_LLM_MODEL` in `.env`.

4. **Optional:** run Hermit’s helper to check connectivity and pull defaults:

   ```bash
   uv run hermit setup
   ```

## Docker

If you use Hermit’s `docker-compose.yml`, Ollama runs in a container; pull models **inside that container** (see the [README](../README.md) Docker section). You do not need a host install of Ollama for that path—only for **native** `uv run hermit` / local API usage.

## More help

- Library & API details: [github.com/ollama/ollama](https://github.com/ollama/ollama)
- Model list: [ollama.com/library](https://ollama.com/library)
