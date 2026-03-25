from __future__ import annotations

from fastapi import APIRouter, Depends

from hermit.api import service as api_service
from hermit.api.dependencies import get_api_settings, get_ingestion_service
from hermit.api.schemas import (
    IngestDirectoryRequest,
    IngestDirectoryResponse,
    IngestFileRequest,
    IngestFileResponse,
)
from hermit.ingestion.service import IngestionService
from hermit.settings import Settings

router = APIRouter(prefix="", tags=["ingestion"])


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
