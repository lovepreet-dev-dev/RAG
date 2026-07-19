"""Document loaders for supported file types.

Each loader reads a file from disk and returns a list of LangChain
``Document`` objects with normalised metadata.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.config.constants import SUPPORTED_EXTENSIONS
from app.utils.helpers import clean_text, get_file_type
from app.utils.logging import get_logger

logger = get_logger(__name__)


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file's raw bytes without loading into memory.

    Streams the file in 64 KB chunks for memory efficiency on large files.

    Args:
        file_path: Absolute path to the file.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    sha = hashlib.sha256()
    with open(file_path, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


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

    # Hash file bytes before loading into memory (optimisation #6)
    file_hash = compute_file_hash(file_path)

    logger.info(
        "loading_document",
        filename=fname,
        type=file_type,
        size=path.stat().st_size,
        file_hash=file_hash[:16],
    )

    loaders = {
        "pdf": _load_pdf,
        "text": _load_text,
        "markdown": _load_text,
        "docx": _load_docx,
        "html": _load_html,
        "csv": _load_csv,
        "json": _load_json,
        "jsonl": _load_jsonl,
    }

    loader_fn = loaders.get(file_type)
    if loader_fn is None:
        raise ValueError(f"No loader for type: {file_type}")

    docs = loader_fn(path, fname)

    # Attach pre-computed hash to every doc for downstream duplicate detection
    for doc in docs:
        doc.metadata["content_hash"] = file_hash

    logger.info("loaded_document", filename=fname, pages=len(docs), content_hash=file_hash[:16])
    return docs


def _load_pdf(path: Path, filename: str) -> list[Document]:
    """Load a PDF file.

    If PyPDFLoader yields no extractable text and OCR is enabled, falls back
    to ``pytesseract`` via ``pdf2image`` for a graceful OCR pass.

    Args:
        path: Path to the PDF.
        filename: Display name.

    Returns:
        List of Documents, one per page.
    """
    from app.config.settings import get_settings
    from langchain_community.document_loaders import PyPDFLoader

    settings = get_settings()
    loader = PyPDFLoader(str(path))
    docs = loader.load()

    # Check if any page has meaningful text
    has_text = any(d.page_content.strip() for d in docs)

    if not has_text and getattr(settings, "enable_ocr", False):
        logger.info("ocr_fallback_triggered", filename=filename, pages=len(docs))
        docs = _ocr_pdf(path, filename, settings)

    for i, doc in enumerate(docs):
        doc.metadata["filename"] = filename
        doc.metadata["page_number"] = i + 1
        doc.metadata["content_type"] = "application/pdf"
        doc.page_content = clean_text(doc.page_content)

    return docs


def _ocr_pdf(path: Path, filename: str, settings: object) -> list[Document]:
    """Perform OCR on a PDF with no extractable text.

    Uses ``pdf2image`` + ``pytesseract``.  Falls back gracefully if the
    libraries are not installed.

    Args:
        path: Path to the PDF.
        filename: Display name.
        settings: Application settings (needs ``ocr_language``).

    Returns:
        List of Documents with OCR text, or an empty list on failure.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        logger.warning(
            "ocr_unavailable",
            filename=filename,
            msg="pdf2image/pytesseract not installed; skipping OCR",
        )
        return []

    try:
        lang = getattr(settings, "ocr_language", "eng")
        images = convert_from_path(str(path))
        docs: list[Document] = []
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang=lang)
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "filename": filename,
                            "page_number": i + 1,
                            "content_type": "application/pdf",
                            "ocr_applied": True,
                        },
                    )
                )
        logger.info("ocr_complete", filename=filename, pages_with_text=len(docs))
        return docs
    except Exception as exc:
        logger.warning("ocr_failed", filename=filename, error=str(exc))
        return []


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

    Large CSVs are processed in configurable batches to limit memory usage.

    Args:
        path: Path to the CSV.
        filename: Display name.

    Returns:
        List of Documents, one per row.
    """
    from app.config.settings import get_settings

    settings = get_settings()
    batch_size = getattr(settings, "csv_batch_size", 500)

    docs: list[Document] = []
    batch_count = 0

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
                        "content_type": "text/csv",
                    },
                )
            )
            # Flush batch periodically
            if len(docs) >= batch_size:
                batch_count += 1
                logger.debug("csv_batch_flushed", batch=batch_count, rows=len(docs))

    if not docs:
        return _load_text(path, filename)

    logger.info("csv_loaded", filename=filename, rows=len(docs), batches=batch_count + 1)
    return docs


def _load_json(path: Path, filename: str) -> list[Document]:
    """Load a JSON file.

    Large JSON arrays are processed in configurable batches.

    Args:
        path: Path to the JSON.
        filename: Display name.

    Returns:
        List of Documents.
    """
    from app.config.settings import get_settings

    settings = get_settings()
    batch_size = getattr(settings, "json_batch_size", 500)

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
                        "content_type": "application/json",
                    },
                )
            )
            # Flush batch periodically
            if len(docs) >= batch_size:
                logger.debug("json_batch_flushed", items=len(docs))

        logger.info("json_loaded", filename=filename, items=len(docs))
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


# Fields tried, in order, when picking the main text of a JSONL record.
_JSONL_TEXT_KEYS = ("text", "content", "body", "abstract", "passage")


def _load_jsonl(path: Path, filename: str) -> list[Document]:
    """Load a JSON-Lines file (one JSON object per line).

    Each line becomes one Document.  The main text is taken from the first
    present key in ``_JSONL_TEXT_KEYS``; a ``title`` is prepended when present,
    and ``id``/``url``/``source``/``title`` are preserved as metadata so
    citations can point back to the original record.  Lines that are not the
    expected object shape fall back to the full JSON dump.

    Args:
        path: Path to the JSONL file.
        filename: Display name.

    Returns:
        List of Documents, one per non-empty line.
    """
    docs: list[Document] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("jsonl_line_skipped", filename=filename, line=line_no)
                continue

            if isinstance(obj, dict):
                text = ""
                for key in _JSONL_TEXT_KEYS:
                    if str(obj.get(key, "")).strip():
                        text = str(obj[key])
                        break
                if not text:
                    text = json.dumps(obj, indent=2)
                title = str(obj.get("title", "")).strip()
                if title:
                    text = f"{title}\n\n{text}"
                metadata = {
                    "filename": filename,
                    "page_number": line_no,
                    "content_type": "application/jsonl",
                    "record_id": obj.get("id", ""),
                    "record_url": obj.get("url", ""),
                    "record_source": obj.get("source", ""),
                    "record_title": title,
                }
            else:
                text = json.dumps(obj, indent=2) if not isinstance(obj, str) else obj
                metadata = {
                    "filename": filename,
                    "page_number": line_no,
                    "content_type": "application/jsonl",
                }

            cleaned = clean_text(text)
            if cleaned:
                docs.append(Document(page_content=cleaned, metadata=metadata))

    if not docs:
        return _load_text(path, filename)

    logger.info("jsonl_loaded", filename=filename, records=len(docs))
    return docs
