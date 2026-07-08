"""Seed the vector store with sample documents for testing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import get_settings
from app.embeddings.factory import get_embeddings_model
from app.ingestion.pipeline import ingest_files
from app.vectorstore.factory import get_vector_store


def main() -> None:
    """Ingest all files from ``data/sample/`` into the vector store."""
    settings = get_settings()
    sample_dir = Path(settings.data_dir) / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_file = sample_dir / "example.txt"
    if not sample_file.exists():
        sample_file.write_text(
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "retrieval from a knowledge base with large language model generation. "
            "RAG systems first retrieve relevant documents based on the user query, "
            "then use those documents as context for the LLM to generate an answer. "
            "This approach reduces hallucination and provides grounded responses."
        )
        print(f"Created sample file: {sample_file}")

    files = [str(f) for f in sample_dir.iterdir() if f.is_file()]
    if not files:
        print("No files found in data/sample/. Add documents and re-run.")
        return

    print(f"Found {len(files)} files to ingest:")
    for f in files:
        print(f"  - {Path(f).name}")

    embeddings = get_embeddings_model(settings)
    vector_store = get_vector_store(embeddings, settings)
    results = ingest_files(files, vector_store, settings)

    print("\nIngestion complete!")
    for r in results:
        print(f"  Document: {r.filename} | Chunks: {r.total_chunks} | Pages: {r.total_pages}")

    print(f"\nTotal documents in store: {vector_store.get_document_count()}")


if __name__ == "__main__":
    main()
