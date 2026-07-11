"""Background job queue for async operations.

Provides an in-process job queue using threading for non-blocking operations
like document ingestion, session cleanup, and index maintenance.
"""

from __future__ import annotations

import enum
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)

_MAX_JOB_HISTORY = 500


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(int, enum.Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass
class BackgroundJob:
    """A background job with status tracking."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_type: str = ""
    func_name: str = ""
    func: Optional[Callable] = field(default=None, repr=False)
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    progress: float = 0.0
    metadata: dict = field(default_factory=dict)


class JobQueue:
    """In-process background job queue with worker threads."""

    def __init__(self, max_workers: int = 2):
        """Initialize the job queue.

        Args:
            max_workers: Number of worker threads.
        """
        self._queue: list[BackgroundJob] = []
        self._jobs: dict[str, BackgroundJob] = {}
        self._workers: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._shutdown = False
        self._max_workers = max_workers

    def start(self) -> None:
        """Start worker threads."""
        if self._workers:
            return
        for i in range(self._max_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"job-worker-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)
        logger.info("job_queue_started", workers=self._max_workers)

    def stop(self) -> None:
        """Signal workers to stop and wait for them."""
        self._shutdown = True
        for t in self._workers:
            t.join(timeout=5)
        self._workers.clear()
        logger.info("job_queue_stopped")

    def submit(
        self,
        func: Callable,
        job_type: str = "generic",
        priority: JobPriority = JobPriority.NORMAL,
        args: tuple = (),
        kwargs: dict | None = None,
        metadata: dict | None = None,
    ) -> BackgroundJob:
        """Submit a function to run in the background.

        Args:
            func: The function to execute.
            job_type: Type label for the job.
            priority: Job priority (higher = processed first).
            args: Positional arguments for the function.
            kwargs: Keyword arguments for the function.
            metadata: Optional metadata dict.

        Returns:
            The BackgroundJob for tracking.
        """
        job = BackgroundJob(
            job_type=job_type,
            func_name=func.__name__,
            func=func,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            metadata=metadata or {},
        )

        with self._lock:
            self._jobs[job.id] = job
            # Insert in priority order (list, not deque — no maxlen crash)
            inserted = False
            for i, existing in enumerate(self._queue):
                if job.priority > existing.priority:
                    self._queue.insert(i, job)
                    inserted = True
                    break
            if not inserted:
                self._queue.append(job)

            # Evict old completed/failed/cancelled jobs to cap memory
            self._evict_old_jobs()

        logger.info(
            "job_submitted",
            job_id=job.id,
            job_type=job_type,
            func=func.__name__,
            priority=priority.value,
        )
        return job

    def get_job(self, job_id: str) -> Optional[BackgroundJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 50,
    ) -> list[BackgroundJob]:
        """List jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        job = self._jobs.get(job_id)
        if not job or job.status != JobStatus.PENDING:
            return False
        job.status = JobStatus.CANCELLED
        with self._lock:
            try:
                self._queue.remove(job)
            except ValueError:
                pass
        logger.info("job_cancelled", job_id=job_id)
        return True

    def _evict_old_jobs(self) -> None:
        """Evict terminal jobs when history exceeds cap."""
        if len(self._jobs) <= _MAX_JOB_HISTORY:
            return
        terminal = (
            j for j in self._jobs.values()
            if j.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
        )
        sorted_terminal = sorted(
            terminal,
            key=lambda j: j.completed_at or j.created_at,
        )
        to_remove = len(sorted_terminal) - (_MAX_JOB_HISTORY // 2)
        for j in sorted_terminal[:to_remove]:
            self._jobs.pop(j.id, None)

    def _worker_loop(self) -> None:
        """Main loop for worker threads."""
        while not self._shutdown:
            job = None
            with self._lock:
                if self._queue:
                    job = self._queue.pop(0)

            if job is None:
                time.sleep(0.1)
                continue

            if job.status == JobStatus.CANCELLED:
                continue

            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            logger.info("job_started", job_id=job.id, func=job.func_name)

            try:
                if job.func is not None and callable(job.func):
                    result = job.func(*job.args, **job.kwargs)
                    job.result = result
                else:
                    job.error = f"Function not callable: {job.func_name}"

                job.status = JobStatus.COMPLETED
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                logger.error(
                    "job_failed",
                    job_id=job.id,
                    func=job.func_name,
                    error=str(exc),
                )
            finally:
                job.completed_at = datetime.now(timezone.utc)
                logger.info(
                    "job_completed",
                    job_id=job.id,
                    status=job.status.value,
                    duration_ms=(
                        (job.completed_at - job.started_at).total_seconds() * 1000
                        if job.started_at and job.completed_at
                        else 0
                    ),
                )


# Global singleton
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get or create the global job queue."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
        _job_queue.start()
    return _job_queue
