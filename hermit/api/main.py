from __future__ import annotations

from fastapi import FastAPI

from hermit.api.routers.collections import router as collections_router
from hermit.api.routers.health import router as health_router
from hermit.api.routers.ingest import router as ingest_router
from hermit.api.routers.query import router as query_router

app = FastAPI(title="Hermit API", version="0.1.0")
app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(collections_router)
