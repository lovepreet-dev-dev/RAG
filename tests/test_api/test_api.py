"""Tests for the API endpoints."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        """Health endpoint should return 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_includes_provider(self, client: TestClient) -> None:
        """Health response should include LLM provider."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "llm_provider" in data
        assert "vector_store" in data


class TestStatsEndpoint:
    """Tests for the stats endpoint."""

    def test_stats_returns_ok(self, client: TestClient) -> None:
        """Stats endpoint should return 200."""
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_queries" in data


class TestUploadEndpoint:
    """Tests for the upload endpoint."""

    def test_upload_txt(self, client: TestClient) -> None:
        """Uploading a TXT file should succeed."""
        content = b"This is a test document for upload."
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", content, "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.txt"

    def test_upload_unsupported_file(self, client: TestClient) -> None:
        """Uploading an unsupported file type should fail."""
        content = b"\x00\x01\x02"
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.bin", content, "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_upload_json(self, client: TestClient) -> None:
        """Uploading a JSON file should succeed."""
        content = json.dumps({"key": "value"}).encode()
        response = client.post(
            "/api/v1/upload",
            files={"file": ("data.json", content, "application/json")},
        )
        assert response.status_code == 200


class TestDocumentsEndpoint:
    """Tests for the documents listing endpoint."""

    def test_list_documents(self, client: TestClient) -> None:
        """Should return a list of documents with pagination."""
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "total_count" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data


class TestDeleteDocumentEndpoint:
    """Tests for document deletion."""

    def test_delete_nonexistent(self, client: TestClient) -> None:
        """Deleting a non-existent document should fail gracefully."""
        response = client.delete("/api/v1/documents/nonexistent-id")
        assert response.status_code in (404, 500)


class TestFeedbackEndpoint:
    """Tests for the feedback endpoint."""

    def test_submit_feedback(self, client: TestClient) -> None:
        """Feedback should be accepted."""
        response = client.post(
            "/api/v1/feedback",
            params={"query_id": "test-query-1", "rating": 5, "comment": "Great answer!"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["rating"] == 5

    def test_invalid_rating(self, client: TestClient) -> None:
        """Rating outside 1-5 should be rejected."""
        response = client.post(
            "/api/v1/feedback",
            params={"query_id": "test-query-1", "rating": 10},
        )
        assert response.status_code == 422
