"""API endpoints for the RAG application.

All request/response models are validated via Pydantic schemas.
Errors return structured JSON with appropriate HTTP status codes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_current_user_optional, get_db, require_role
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

    Supports both synchronous and background ingestion. When ``background=true``,
    the job is queued and returns immediately with a job ID.

    Returns:
        IngestResponse with document summaries or job info.

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

    # Background ingestion
    if getattr(request, "background", False):
        from app.jobs import get_job_queue, JobPriority
        from app.jobs.tasks import ingest_documents_background

        queue = get_job_queue()
        job = queue.submit(
            func=lambda: ingest_documents_background(
                file_paths=file_paths,
                chunking_strategy=request.chunking_strategy.value if request.chunking_strategy else None,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
            ),
            job_type="ingestion",
            priority=JobPriority.HIGH,
            metadata={"filenames": request.filenames},
        )
        return IngestResponse(
            documents=[],
            total_chunks=0,
            message=f"Ingestion job queued (job_id={job.id}). Check /api/v1/jobs/{job.id} for status.",
        )

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
    db: Session = Depends(get_db),
) -> Union[QueryResponse, StreamingResponse]:
    """Conversational endpoint with memory and optional streaming.

    Messages are persisted to the database for long-term retention.
    When a session_id is provided, history is loaded from the database
    instead of the in-memory cache.

    Set ``stream=true`` in the request to receive Server-Sent Events.

    Returns:
        QueryResponse or StreamingResponse depending on ``stream`` flag.

    Raises:
        HTTPException: On chat failure.
    """
    from app.memory.chat_store import (
        get_or_create_session,
        load_session_history,
        persist_chat_response,
        persist_message,
    )

    service = get_rag_service()

    # Resolve session: use DB-backed sessions when user is authenticated
    session_id = request.session_id or generate_id()
    request.session_id = session_id  # Ensure service uses the same session ID
    use_db = current_user is not None

    if use_db:
        get_or_create_session(
            db=db,
            session_id=session_id,
            user_id=current_user.id,
            title=request.message[:80],
        )
        # Load history from database instead of in-memory dict
        db_history = load_session_history(db=db, session_id=session_id)
        # Temporarily inject DB history into the service's session cache
        service._sessions[session_id] = db_history

    if request.stream:
        async def event_generator():  # type: ignore[no-untyped-def]
            final_chunk_data = {}
            try:
                async for chunk in service.chat_stream(request):
                    data = chunk.model_dump_json()
                    yield f"data: {data}\n\n"
                    if chunk.is_final:
                        final_chunk_data = {
                            "query_id": chunk.query_id,
                            "sources": chunk.sources,
                            "metadata": chunk.metadata,
                        }
                yield "data: [DONE]\n\n"

                # Persist messages with metadata after stream completes
                if use_db:
                    persist_message(
                        db=db,
                        session_id=session_id,
                        role="user",
                        content=request.message,
                    )
                    # Get the full answer from the service's session history
                    history = service._sessions.get(session_id, [])
                    if history and history[-1]["role"] == "assistant":
                        meta = final_chunk_data.get("metadata", {})
                        persist_message(
                            db=db,
                            session_id=session_id,
                            role="assistant",
                            content=history[-1]["content"],
                            query_id=final_chunk_data.get("query_id"),
                            sources=[s.model_dump() for s in final_chunk_data["sources"]] if final_chunk_data.get("sources") else None,
                            confidence_score=meta.get("retrieval_confidence"),
                            model_used=meta.get("model"),
                        )
            except Exception as exc:
                logger.error("streaming_chat_failed", error=str(exc))
                import json

                error_payload = json.dumps({"error": str(exc), "is_final": True})
                yield f"data: {error_payload}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    try:
        response = service.chat(request)

        # Persist to database
        if use_db:
            persist_chat_response(
                db=db,
                session_id=session_id,
                user_message=request.message,
                response=response,
            )

        return response
    except Exception as exc:
        logger.error("chat_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}")


@router.get(
    "/documents",
    summary="List indexed documents",
    responses={500: {"model": ErrorResponse}},
)
def list_documents(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=50, ge=1, le=200, description="Items per page"),
) -> dict:  # type: ignore[type-arg]
    """Return a summary of all indexed documents with pagination.

    Returns:
        Dict with ``documents`` list, ``total_count``, and pagination info.
    """
    service = get_rag_service()
    try:
        docs = service.get_documents()
        total = len(docs)
        total_pages = max(1, (total + page_size - 1) // page_size)
        start = (page - 1) * page_size
        end = start + page_size
        page_docs = docs[start:end]

        return {
            "data": [d.model_dump() for d in page_docs],
            "total_count": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        }
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
    "/documents/{document_id}/content",
    summary="Get document content",
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def get_document_content(
    document_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Get the full text content of a document, split into pages/chunks.

    Returns:
        Dict with document_id, filename, and pages list.
    """
    service = get_rag_service()
    try:
        chunks = service.vector_store.get_document_content(document_id)
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{document_id}' not found or has no content.",
            )

        # Group chunks by page
        pages: dict[int, list[str]] = {}
        filename = ""
        for chunk in chunks:
            page = chunk.metadata.get("page_number", 0)
            if page not in pages:
                pages[page] = []
            pages[page].append(chunk.page_content)
            if not filename:
                filename = chunk.metadata.get("filename", "")

        # Combine chunks per page
        page_contents = []
        for page_num in sorted(pages.keys()):
            page_contents.append({
                "page_number": page_num,
                "content": "\n\n".join(pages[page_num]),
            })

        return {
            "document_id": document_id,
            "filename": filename,
            "total_pages": len(page_contents),
            "total_chunks": len(chunks),
            "pages": page_contents,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_document_content_failed", document_id=document_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to get document content: {exc}")


@router.get(
    "/documents/{document_id}/download",
    summary="Download original document file",
    responses={404: {"model": ErrorResponse}},
)
def download_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
):
    """Download the original uploaded file.

    Returns:
        File response with the original document.
    """
    from fastapi.responses import FileResponse

    service = get_rag_service()
    try:
        # Get filename from vector store
        chunks = service.vector_store.get_document_content(document_id)
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{document_id}' not found.",
            )

        filename = chunks[0].metadata.get("filename", "unknown")
        upload_dir = _get_upload_dir()
        file_path = (upload_dir / filename).resolve()

        # Prevent path traversal
        if not str(file_path).startswith(str(upload_dir.resolve())):
            raise HTTPException(
                status_code=400,
                detail="Invalid file path",
            )

        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Original file '{filename}' not found on disk.",
            )

        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/octet-stream",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("download_failed", document_id=document_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Download failed: {exc}")


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
    from app.utils.cache import get_cache

    cache = get_cache()
    cached_stats = cache.get("stats")
    if cached_stats:
        return StatsResponse(**cached_stats)

    service = get_rag_service()
    stats = service.get_stats()
    cache.set("stats", stats.model_dump(), ttl=30)
    return stats


@router.post(
    "/feedback",
    summary="Submit feedback on a response",
)
def submit_feedback(
    query_id: str = Query(..., min_length=1),
    rating: int = Query(..., ge=1, le=5),
    comment: Optional[str] = Query(default=None, max_length=2000),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Record user feedback for a specific query response.

    Returns:
        Confirmation dict with query_id and rating.
    """
    from app.models.user import Feedback

    feedback = Feedback(
        query_id=query_id,
        user_id=current_user.id if current_user else None,
        rating=rating,
        comment=comment,
    )
    db.add(feedback)
    db.commit()

    logger.info(
        "feedback_received",
        query_id=query_id,
        rating=rating,
        comment=comment,
        user_id=current_user.id if current_user else None,
    )
    return {"message": "Feedback recorded.", "query_id": query_id, "rating": rating}


# ── Background Job Endpoints ────────────────────────────────────────────


@router.get(
    "/jobs/{job_id}",
    summary="Get background job status",
    responses={404: {"model": ErrorResponse}},
)
def get_job_status(
    job_id: str,
    current_user: User = Depends(require_role(Role.ADMIN)),
) -> dict:
    """Get the status of a background job."""
    from app.jobs import get_job_queue

    queue = get_job_queue()
    job = queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return {
        "id": job.id,
        "job_type": job.job_type,
        "status": job.status.value,
        "priority": job.priority.value,
        "progress": job.progress,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "result": job.result,
        "error": job.error,
        "metadata": job.metadata,
    }


@router.get(
    "/jobs",
    summary="List background jobs",
)
def list_jobs(
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(require_role(Role.ADMIN)),
) -> dict:
    """List background jobs, optionally filtered by status."""
    from app.jobs import get_job_queue, JobStatus

    queue = get_job_queue()
    job_status = JobStatus(status) if status else None
    jobs = queue.list_jobs(status=job_status, limit=limit)

    return {
        "jobs": [
            {
                "id": j.id,
                "job_type": j.job_type,
                "status": j.status.value,
                "created_at": j.created_at.isoformat(),
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            }
            for j in jobs
        ],
        "total_count": len(jobs),
    }


@router.post(
    "/jobs/{job_id}/cancel",
    responses={404: {"model": ErrorResponse}},
)
def cancel_job(
    job_id: str,
    current_user: User = Depends(require_role(Role.ADMIN)),
) -> dict:
    """Cancel a pending background job."""
    from app.jobs import get_job_queue

    queue = get_job_queue()
    success = queue.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found or not cancellable",
        )
    return {"message": f"Job '{job_id}' cancelled", "job_id": job_id}
