from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from hermit.api.dependencies import get_ingestion_service
from hermit.ingestion.service import IngestionService

router = APIRouter(prefix="", tags=["ingestion"])


class IngestFileRequest(BaseModel):
    path: str


class IngestDirectoryRequest(BaseModel):
    path: str
    recursive: bool | None = None


@router.post("/ingest")
def ingest_file(
    request: IngestFileRequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> dict[str, object]:
    result = ingestion_service.ingest_file(Path(request.path))
    source = result.processed_sources[0] if result.processed_sources else request.path
    return {
        "status": "ok",
        "chunks_added": result.total_chunks,
        "source": source,
    }


@router.post("/ingest/directory")
def ingest_directory(
    request: IngestDirectoryRequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> dict[str, object]:
    result = ingestion_service.ingest_directory(Path(request.path), recursive=request.recursive)
    return {
        "status": "ok",
        "files_processed": result.files_processed,
        "total_chunks": result.total_chunks,
    }
