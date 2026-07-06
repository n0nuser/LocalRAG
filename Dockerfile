FROM python:3.13-slim AS base

WORKDIR /app

# tesseract-ocr: required at runtime for scanned/image-only PDF pages (see docs/ocr.md).
RUN apt-get update \
    && apt-get install --no-install-recommends -y tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md /app/

# Install all external dependencies; skip building the local package (no source yet).
# The dev override (docker-compose.override.yml) mounts the live source and stops here.
RUN uv sync --no-dev --no-install-project

FROM base

COPY localrag /app/localrag

# Build and install the local package on top of the already-cached deps.
RUN uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "localrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
