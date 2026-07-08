"""API endpoints for the RAG application.

All request/response models are validated via Pydantic schemas.
Errors return structured JSON with appropriate HTTP status codes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.api.deps import get_current_user, get_current_user_optional, require_role
from app.models.schemas import (
    ChatRequest,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
    UploadResponse,
)
from app.models.user import Role, User
from app.services.dependencies import get_rag_service
from app.utils.helpers import is_supported_file, safe_filename
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _get_upload_dir() -> Path:
    """Return the upload directory, creating it if needed.

    Returns:
        Path to the upload directory.
    """
    from app.config.settings import get_settings

    upload_dir = Path(get_settings().upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a document",
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(require_role(Role.ADMIN, Role.EMPLOYEE)),
) -> UploadResponse:
    """Upload a document file for later ingestion.

    Supported formats: PDF, TXT, DOCX, Markdown, HTML, CSV, JSON.

    Returns:
        UploadResponse with filename, size, and message.

    Raises:
        HTTPException: On missing filename, unsupported type, oversized file, or write failure.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    if not is_supported_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Supported: PDF, TXT, DOCX, MD, HTML, CSV, JSON",
        )

    try:
        content = await file.read()
    except Exception as exc:
        logger.error("upload_read_failed", filename=file.filename, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to read file: {exc}")

    from app.config.settings import get_settings

    settings = get_settings()
    max_bytes = settings.max_upload_bytes()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb} MB.",
        )

    upload_dir = _get_upload_dir()
    safe_name = safe_filename(file.filename)
    dest = upload_dir / safe_name
    try:
        dest.write_bytes(content)
    except Exception as exc:
        logger.error("upload_save_failed", filename=safe_name, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    logger.info("file_uploaded", filename=safe_name, size=len(content))
    return UploadResponse(
        filename=safe_name,
        file_size=len(content),
        message=f"File '{safe_name}' uploaded successfully.",
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documents into the vector store",
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def ingest_documents(
    request: IngestRequest,
    current_user: User = Depends(require_role(Role.ADMIN)),
) -> IngestResponse:
    """Ingest uploaded files: extract text, chunk, embed, and store.

    Returns:
        IngestResponse with document summaries.

    Raises:
        HTTPException: On missing files or ingestion failure.
    """
    service = get_rag_service()
    upload_dir = _get_upload_dir()
    file_paths = [str(upload_dir / f) for f in request.filenames]

    missing = [fp for fp in file_paths if not Path(fp).exists()]
    if missing:
        names = [Path(p).name for p in missing]
        raise HTTPException(status_code=400, detail=f"Files not found: {names}")

    try:
        return service.upload_and_ingest(
            file_paths=file_paths,
            chunking_strategy=(
                request.chunking_strategy.value if request.chunking_strategy else None
            ),
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
    except Exception as exc:
        logger.error("ingestion_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG system",
    responses={500: {"model": ErrorResponse}},
)
def query_rag(
    request: QueryRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> QueryResponse:
    """Ask a question and get an answer with sources.

    Returns:
        QueryResponse with answer, sources, and metadata.

    Raises:
        HTTPException: On query failure.
    """
    service = get_rag_service()
    try:
        return service.query(request)
    except Exception as exc:
        logger.error("query_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")


@router.post(
    "/chat",
    response_model=None,
    summary="Chat with the RAG system (supports streaming)",
    responses={500: {"model": ErrorResponse}},
)
async def chat_rag(
    request: ChatRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> Union[QueryResponse, StreamingResponse]:
    """Conversational endpoint with memory and optional streaming.

    Set ``stream=true`` in the request to receive Server-Sent Events.

    Returns:
        QueryResponse or StreamingResponse depending on ``stream`` flag.

    Raises:
        HTTPException: On chat failure.
    """
    service = get_rag_service()

    if request.stream:
        async def event_generator():  # type: ignore[no-untyped-def]
            try:
                async for chunk in service.chat_stream(request):
                    data = chunk.model_dump_json()
                    yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error("streaming_chat_failed", error=str(exc))
                import json

                error_payload = json.dumps({"error": str(exc), "is_final": True})
                yield f"data: {error_payload}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    try:
        return service.chat(request)
    except Exception as exc:
        logger.error("chat_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}")


@router.get(
    "/documents",
    summary="List indexed documents",
    responses={500: {"model": ErrorResponse}},
)
def list_documents() -> dict:  # type: ignore[type-arg]
    """Return a summary of all indexed documents.

    Returns:
        Dict with ``documents`` list and ``total_count``.
    """
    service = get_rag_service()
    try:
        docs = service.get_documents()
        return {"documents": [d.model_dump() for d in docs], "total_count": len(docs)}
    except Exception as exc:
        logger.error("list_documents_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {exc}")


@router.delete(
    "/documents/{document_id}",
    summary="Delete a document by ID",
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def delete_document(
    document_id: str,
    current_user: User = Depends(require_role(Role.ADMIN)),
) -> dict[str, str]:
    """Delete a document and all its chunks from the vector store.

    Returns:
        Confirmation dict with document_id.

    Raises:
        HTTPException: 404 if not found, 500 on failure.
    """
    service = get_rag_service()
    try:
        success = service.delete_document(document_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{document_id}' not found or could not be deleted.",
            )
        return {"message": f"Document '{document_id}' deleted.", "document_id": document_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("delete_failed", document_id=document_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
def health_check() -> HealthResponse:
    """Return system health status.

    Returns:
        HealthResponse with component status.
    """
    service = get_rag_service()
    return service.get_health()


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="System statistics",
)
def get_stats() -> StatsResponse:
    """Return system usage statistics.

    Returns:
        StatsResponse with usage metrics.
    """
    service = get_rag_service()
    return service.get_stats()


@router.post(
    "/feedback",
    summary="Submit feedback on a response",
)
def submit_feedback(
    query_id: str = Query(..., min_length=1),
    rating: int = Query(..., ge=1, le=5),
    comment: Optional[str] = Query(default=None, max_length=2000),
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> dict[str, object]:
    """Record user feedback for a specific query response.

    Returns:
        Confirmation dict with query_id and rating.
    """
    logger.info(
        "feedback_received",
        query_id=query_id,
        rating=rating,
        comment=comment,
    )
    return {"message": "Feedback recorded.", "query_id": query_id, "rating": rating}
