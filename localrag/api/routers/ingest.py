from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from localrag.api import service as api_service
from localrag.api.dependencies import get_api_settings, get_ingestion_service, require_api_key
from localrag.api.schemas import (
    IngestDirectoryRequest,
    IngestDirectoryResponse,
    IngestFileRequest,
    IngestFileResponse,
)
from localrag.ingestion.service import IngestionService
from localrag.settings import Settings

router = APIRouter(prefix="", tags=["ingestion"], dependencies=[Depends(require_api_key)])

_UPLOAD_DESCRIPTION = """
Ingest a file selected from the caller's own filesystem (browser "Choose File"
dialog), instead of a server-side path like `POST /ingest` requires.

**Limitations:**
- **Size cap:** enforced by counting streamed bytes against `UPLOAD_MAX_BYTES`
  (default 100 MB), not by trusting the `Content-Length` header. Exceeding it
  aborts the upload and deletes the partial file — no partial ingest.
- **Extension allow-list only:** the file extension must be one of the types
  `localrag/ingestion/loader.py` knows how to parse (pdf, docx, md, txt, code
  files). The `Content-Type` header sent by the browser is **not** checked —
  a mismatched or spoofed MIME type on an allowed extension will still be
  parsed as that extension.
- **No malware/antivirus scanning.** Uploaded content is trusted the same way
  a server-side path would be; do not expose this endpoint to untrusted
  callers without a scanning layer in front of it.
- **Persistent storage, not transient:** the file is saved under
  `UPLOAD_DIR` (bypassing `INGEST_ROOTS`, since the server — not the
  caller — chooses the destination) and stays there so
  `POST /collections/rebuild` can re-embed it later. Deleting it from disk
  breaks rebuild for that source.
- **Filename collisions** are resolved by appending a random suffix; check
  the returned `source` field for the actual stored path.
- **Single file per request** — no batch/zip upload.
- **One automatic retry:** if parsing/embedding fails transiently, it's retried
  once before giving up. A failure that persists after the retry is returned
  as `502 Bad Gateway` (not silently swallowed) and logged server-side.
- Same `X-API-Key` requirement as the other ingest endpoints.
"""


@router.post(
    "/ingest/upload",
    response_model=IngestFileResponse,
    summary="Upload and ingest a file",
    description=_UPLOAD_DESCRIPTION,
)
def ingest_upload(
    file: UploadFile = File(..., description="Document to ingest."),
    embed_model: str | None = Form(
        default=None,
        description="Override OLLAMA_EMBED_MODEL for this request.",
    ),
    settings: Settings = Depends(get_api_settings),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> IngestFileResponse:
    return api_service.ingest_upload(
        file_name=file.filename or "upload",
        file_obj=file.file,
        embed_model=embed_model,
        settings=settings,
        ingestion_service=ingestion_service,
    )


@router.post("/ingest", response_model=IngestFileResponse)
def ingest_file(
    request: IngestFileRequest,
    settings: Settings = Depends(get_api_settings),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> IngestFileResponse:
    return api_service.ingest_file(request, settings, ingestion_service)


@router.post("/ingest/directory", response_model=IngestDirectoryResponse)
def ingest_directory(
    request: IngestDirectoryRequest,
    settings: Settings = Depends(get_api_settings),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> IngestDirectoryResponse:
    return api_service.ingest_directory(request, settings, ingestion_service)
