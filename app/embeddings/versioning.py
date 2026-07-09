"""Embedding versioning and index metadata tracking.

Tracks the embedding model, version, dimensions, and index version
used when building a vector store.  Supports reindex detection when
the embedding model changes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingMetadata:
    """Metadata about the embeddings used to build an index."""

    model_name: str = ""
    model_version: str = ""
    dimensions: int = 0
    index_version: str = "1.0.0"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    chunk_count: int = 0
    document_count: int = 0
    checksum: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingMetadata:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class EmbeddingVersionTracker:
    """Tracks embedding versions and detects when reindexing is needed.

    Persists metadata to disk alongside the vector store index.
    Compares current settings against stored metadata to determine
    if a full reindex is required.
    """

    def __init__(self, persist_directory: str | Path | None = None) -> None:
        """Initialize the version tracker.

        Args:
            persist_directory: Directory where the index is persisted.
        """
        self._persist_directory = Path(persist_directory) if persist_directory else None
        self._metadata: EmbeddingMetadata = EmbeddingMetadata()
        self._loaded = False

    @property
    def metadata(self) -> EmbeddingMetadata:
        """Return the current embedding metadata."""
        return self._metadata

    def load(self) -> bool:
        """Load persisted embedding metadata.

        Returns:
            True if metadata was loaded successfully.
        """
        if self._persist_directory is None:
            return False

        meta_path = self._persist_directory / "embedding_metadata.json"
        if not meta_path.exists():
            logger.debug("no_embedding_metadata", path=str(meta_path))
            return False

        try:
            data = json.loads(meta_path.read_text())
            self._metadata = EmbeddingMetadata.from_dict(data)
            self._loaded = True
            logger.info(
                "embedding_metadata_loaded",
                model=self._metadata.model_name,
                version=self._metadata.model_version,
                dimensions=self._metadata.dimensions,
            )
            return True
        except Exception as exc:
            logger.warning("embedding_metadata_load_failed", error=str(exc))
            return False

    def save(self) -> None:
        """Persist embedding metadata to disk."""
        if self._persist_directory is None:
            return

        self._persist_directory.mkdir(parents=True, exist_ok=True)
        meta_path = self._persist_directory / "embedding_metadata.json"
        meta_path.write_text(json.dumps(self._metadata.to_dict(), indent=2))
        logger.debug("embedding_metadata_saved", path=str(meta_path))

    def update(
        self,
        model_name: str,
        model_version: str = "",
        dimensions: int = 0,
        chunk_count: int = 0,
        document_count: int = 0,
    ) -> None:
        """Update the embedding metadata.

        Args:
            model_name: Name of the embedding model.
            model_version: Version string of the model.
            dimensions: Embedding vector dimensions.
            chunk_count: Number of chunks in the index.
            document_count: Number of documents in the index.
        """
        self._metadata = EmbeddingMetadata(
            model_name=model_name,
            model_version=model_version,
            dimensions=dimensions,
            index_version=self._metadata.index_version,
            created_at=datetime.now(timezone.utc).isoformat(),
            chunk_count=chunk_count,
            document_count=document_count,
        )
        self.save()

    def needs_reindex(
        self,
        current_model: str,
        current_dimensions: int = 0,
    ) -> tuple[bool, str]:
        """Check if the current embedding model differs from the stored one.

        Args:
            current_model: The currently configured embedding model.
            current_dimensions: The currently configured embedding dimensions.

        Returns:
            Tuple of (needs_reindex, reason).
        """
        if not self._loaded:
            self.load()

        stored = self._metadata

        if not stored.model_name:
            return False, "No stored metadata — first run"

        if stored.model_name != current_model:
            return True, (
                f"Model changed: '{stored.model_name}' -> '{current_model}'"
            )

        if current_dimensions > 0 and stored.dimensions != current_dimensions:
            return True, (
                f"Dimensions changed: {stored.dimensions} -> {current_dimensions}"
            )

        return False, "Embedding metadata matches"
