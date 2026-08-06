from __future__ import annotations

import tempfile
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import ORJSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from hybrid_rag import __version__
from hybrid_rag.api.schemas import AskRequest, DeleteResponse, HealthResponse
from hybrid_rag.container import Components, build_components
from hybrid_rag.models import AnswerResult, DocumentSummary, IngestionResult


def create_app(component_factory: Callable[[], Components] = build_components) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.components = await anyio.to_thread.run_sync(component_factory)
        yield

    application = FastAPI(
        title="Hybrid Docs RAG API",
        version=__version__,
        description=(
            "Hybrid dense + BM25 retrieval, RRF fusion, reranking, grounded generation, "
            "citation verification, and confidence scoring."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_id_middleware(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        response.headers["x-content-type-options"] = "nosniff"
        return response

    @application.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health(request: Request) -> HealthResponse:
        components: Components = request.app.state.components
        chunks, _ = await anyio.to_thread.run_sync(components.vector_store.all)
        return HealthResponse(
            status="ok",
            indexed_chunks=len(chunks),
            embedding_provider=components.embeddings.name,
            generation_provider=components.generator.name,
            vector_backend=components.settings.vector_backend,
        )

    @application.post("/v1/ask", response_model=AnswerResult, tags=["query"])
    async def ask(payload: AskRequest, request: Request) -> AnswerResult:
        components: Components = request.app.state.components
        return await anyio.to_thread.run_sync(
            lambda: components.service.ask(payload.question, mode=payload.retrieval_mode)
        )

    @application.get("/v1/documents", response_model=list[DocumentSummary], tags=["documents"])
    async def list_documents(request: Request) -> list[DocumentSummary]:
        components: Components = request.app.state.components
        return await anyio.to_thread.run_sync(components.service.list_documents)

    @application.post(
        "/v1/ingest",
        response_model=IngestionResult,
        status_code=status.HTTP_201_CREATED,
        tags=["documents"],
    )
    async def ingest(
        request: Request,
        file: UploadFile = File(...),
        strategy: str | None = Query(default=None, pattern="^(fixed|recursive|semantic)$"),
    ) -> IngestionResult:
        components: Components = request.app.state.components
        suffix = Path(file.filename or "upload").suffix.lower()
        if suffix not in components.settings.allowed_extensions:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}")
        max_bytes = components.settings.max_upload_mb * 1024 * 1024
        content = await file.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail="Upload exceeds configured size limit")
        safe_name = Path(file.filename or f"upload{suffix}").name
        with tempfile.TemporaryDirectory(prefix="hybrid-rag-") as temp_dir:
            path = Path(temp_dir) / safe_name
            path.write_bytes(content)
            return await anyio.to_thread.run_sync(
                lambda: components.ingestion.ingest(path, strategy=strategy)
            )

    @application.delete(
        "/v1/documents/{document_id}", response_model=DeleteResponse, tags=["documents"]
    )
    async def delete_document(document_id: str, request: Request) -> DeleteResponse:
        components: Components = request.app.state.components

        def remove() -> int:
            count = components.vector_store.delete_document(document_id)
            components.ingestion.synchronize_indexes()
            return count

        deleted = await anyio.to_thread.run_sync(remove)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        return DeleteResponse(document_id=document_id, chunks_deleted=deleted)

    return application


app = create_app()
