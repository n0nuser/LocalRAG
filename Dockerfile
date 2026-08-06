FROM python:3.13-slim@sha256:6771159cd4fa5d9bba1258caf0b82e6b73458c694d178ad97c5e925c2d0e1a91 AS base

ARG LOCALRAG_BUILD_SHA=unknown
ENV LOCALRAG_BUILD_SHA=${LOCALRAG_BUILD_SHA}

WORKDIR /app

# tesseract-ocr: required at runtime for scanned/image-only PDF pages (see docs/ocr.md).
# tesseract-ocr-spa: Spanish language pack (set OCR_LANGUAGE=spa to use it).
RUN apt-get update \
    && apt-get install --no-install-recommends -y tesseract-ocr tesseract-ocr-spa \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.12.1

COPY pyproject.toml uv.lock README.md /app/

# Install all external dependencies; skip building the local package (no source yet).
# The dev override (docker-compose.override.yml) mounts the live source and stops here.
RUN uv sync --locked --no-dev --no-install-project

FROM base AS app

COPY localrag /app/localrag
RUN useradd --create-home --uid 10001 localrag \
    && mkdir -p /app/data \
    && chown -R localrag:localrag /app

# Build and install the local package on top of the already-cached deps.
RUN uv sync --locked --no-dev \
    && chown -R localrag:localrag /app/.venv

USER localrag

EXPOSE 8000 8002

CMD ["/app/.venv/bin/uvicorn", "localrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM app AS benchmark

# The benchmark image is self-contained: runner, registered fixtures, schemas,
# failure-artifact code, and the committed dependency/model lock identities.
COPY --chown=localrag:localrag evals /app/evals
COPY --chown=localrag:localrag scripts /app/scripts
COPY --chown=localrag:localrag docker /app/docker
RUN mkdir -p /app/evals/results/docker/matrices \
    && chown -R localrag:localrag /app/evals/results
