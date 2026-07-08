"""Tests for the document ingestion pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ingestion.loaders import load_document
from app.utils.helpers import clean_text, generate_content_hash, is_supported_file


class TestDocumentLoaders:
    """Tests for document loading."""

    def test_load_txt(self, tmp_path: Path) -> None:
        """Plain text files should load correctly."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is a test document.\nIt has multiple lines.")
        docs = load_document(str(txt_file), "test.txt")
        assert len(docs) >= 1
        assert "test document" in docs[0].page_content

    def test_load_json(self, tmp_path: Path) -> None:
        """JSON files should be loaded as documents."""
        json_file = tmp_path / "test.json"
        data = {"key": "value", "items": [1, 2, 3]}
        json_file.write_text(json.dumps(data))
        docs = load_document(str(json_file), "test.json")
        assert len(docs) >= 1
        assert "key" in docs[0].page_content

    def test_load_json_array(self, tmp_path: Path) -> None:
        """JSON arrays should create one document per item."""
        json_file = tmp_path / "array.json"
        data = [{"id": 1, "text": "first"}, {"id": 2, "text": "second"}]
        json_file.write_text(json.dumps(data))
        docs = load_document(str(json_file), "array.json")
        assert len(docs) == 2

    def test_load_csv(self, tmp_path: Path) -> None:
        """CSV files should create documents from rows."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25")
        docs = load_document(str(csv_file), "test.csv")
        assert len(docs) >= 2

    def test_load_unsupported(self, tmp_path: Path) -> None:
        """Unsupported file types should raise ValueError."""
        bin_file = tmp_path / "test.bin"
        bin_file.write_bytes(b"\x00\x01\x02")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(str(bin_file), "test.bin")

    def test_load_nonexistent(self) -> None:
        """Non-existent files should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_document("/nonexistent/file.txt", "file.txt")

    def test_markdown_loading(self, tmp_path: Path) -> None:
        """Markdown files should load correctly."""
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Title\n\n## Section\n\nSome content here.")
        docs = load_document(str(md_file), "readme.md")
        assert len(docs) >= 1
        assert "Title" in docs[0].page_content


class TestHelpers:
    """Tests for utility helper functions."""

    def test_clean_text(self) -> None:
        """Text should be cleaned of excess whitespace."""
        dirty = "Hello   world\n\n\n\n\n  extra   spaces  "
        cleaned = clean_text(dirty)
        assert "Hello world" in cleaned
        assert "\n\n\n\n" not in cleaned

    def test_generate_content_hash(self) -> None:
        """Same content should produce the same hash."""
        h1 = generate_content_hash("hello world")
        h2 = generate_content_hash("hello world")
        assert h1 == h2

    def test_different_content_different_hash(self) -> None:
        """Different content should produce different hashes."""
        h1 = generate_content_hash("hello")
        h2 = generate_content_hash("world")
        assert h1 != h2

    def test_is_supported_file(self) -> None:
        """Supported extensions should be recognised."""
        assert is_supported_file("doc.pdf")
        assert is_supported_file("notes.txt")
        assert is_supported_file("readme.md")
        assert is_supported_file("data.csv")
        assert not is_supported_file("image.png")
        assert not is_supported_file("archive.zip")
