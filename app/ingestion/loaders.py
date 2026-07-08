"""Document loaders for supported file types.

Each loader reads a file from disk and returns a list of LangChain
``Document`` objects with normalised metadata.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.config.constants import SUPPORTED_EXTENSIONS
from app.utils.helpers import clean_text, get_file_type
from app.utils.logging import get_logger

logger = get_logger(__name__)


def load_document(file_path: str, filename: Optional[str] = None) -> list[Document]:
    """Load a document from disk and return LangChain ``Document`` objects.

    Args:
        file_path: Absolute path to the file.
        filename: Optional override for the filename metadata.

    Returns:
        List of ``Document`` objects (one per page/section).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is not supported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    fname = filename or path.name
    file_type = get_file_type(fname)
    if file_type is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    logger.info(
        "loading_document",
        filename=fname,
        type=file_type,
        size=path.stat().st_size,
    )

    loaders = {
        "pdf": _load_pdf,
        "text": _load_text,
        "markdown": _load_text,
        "docx": _load_docx,
        "html": _load_html,
        "csv": _load_csv,
        "json": _load_json,
    }

    loader_fn = loaders.get(file_type)
    if loader_fn is None:
        raise ValueError(f"No loader for type: {file_type}")

    docs = loader_fn(path, fname)
    logger.info("loaded_document", filename=fname, pages=len(docs))
    return docs


def _load_pdf(path: Path, filename: str) -> list[Document]:
    """Load a PDF file.

    Args:
        path: Path to the PDF.
        filename: Display name.

    Returns:
        List of Documents, one per page.
    """
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(path))
    docs = loader.load()
    for i, doc in enumerate(docs):
        doc.metadata["filename"] = filename
        doc.metadata["page_number"] = i + 1
        doc.metadata["content_type"] = "application/pdf"
        doc.page_content = clean_text(doc.page_content)
    return docs


def _load_text(path: Path, filename: str) -> list[Document]:
    """Load a plain text or markdown file.

    Args:
        path: Path to the file.
        filename: Display name.

    Returns:
        Single-element list with the full document.
    """
    content = path.read_text(encoding="utf-8", errors="ignore")
    content = clean_text(content)
    if not content:
        return []
    return [
        Document(
            page_content=content,
            metadata={
                "filename": filename,
                "page_number": 1,
                "content_type": "text/plain",
            },
        )
    ]


def _load_docx(path: Path, filename: str) -> list[Document]:
    """Load a DOCX file.

    Args:
        path: Path to the DOCX.
        filename: Display name.

    Returns:
        List of Documents.
    """
    from langchain_community.document_loaders import Docx2txtLoader

    loader = Docx2txtLoader(str(path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["filename"] = filename
        doc.metadata["page_number"] = 1
        doc.metadata["content_type"] = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        doc.page_content = clean_text(doc.page_content)
    return docs


def _load_html(path: Path, filename: str) -> list[Document]:
    """Load an HTML file.

    Args:
        path: Path to the HTML.
        filename: Display name.

    Returns:
        List of Documents.
    """
    from langchain_community.document_loaders import BSHTMLLoader

    loader = BSHTMLLoader(str(path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["filename"] = filename
        doc.metadata["page_number"] = 1
        doc.metadata["content_type"] = "text/html"
        doc.page_content = clean_text(doc.page_content)
    return docs


def _load_csv(path: Path, filename: str) -> list[Document]:
    """Load a CSV file, converting each row to a Document.

    Args:
        path: Path to the CSV.
        filename: Display name.

    Returns:
        List of Documents, one per row.
    """
    docs: list[Document] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            content = json.dumps(row, indent=2)
            docs.append(
                Document(
                    page_content=clean_text(content),
                    metadata={
                        "filename": filename,
                        "page_number": 1,
                        "chunk_id": i,
                        "content_type": "text/csv",
                    },
                )
            )
    return docs if docs else _load_text(path, filename)


def _load_json(path: Path, filename: str) -> list[Document]:
    """Load a JSON file.

    Args:
        path: Path to the JSON.
        filename: Display name.

    Returns:
        List of Documents.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _load_text(path, filename)

    if isinstance(data, list):
        docs: list[Document] = []
        for i, item in enumerate(data):
            content = json.dumps(item, indent=2) if not isinstance(item, str) else item
            docs.append(
                Document(
                    page_content=clean_text(content),
                    metadata={
                        "filename": filename,
                        "page_number": 1,
                        "chunk_id": i,
                        "content_type": "application/json",
                    },
                )
            )
        return docs

    content = json.dumps(data, indent=2) if not isinstance(data, str) else data
    return [
        Document(
            page_content=clean_text(content),
            metadata={
                "filename": filename,
                "page_number": 1,
                "content_type": "application/json",
            },
        )
    ]
