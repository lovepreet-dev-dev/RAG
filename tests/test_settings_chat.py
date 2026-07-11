"""Tests for settings and chat history endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, StaticPool
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, get_db
from app.models.user import User, Role, RefreshToken, ChatSession, ChatMessage, Feedback, AuditLog, SystemSetting, UploadedDocument  # noqa: F401
from app.services.auth_service import create_user
from app.api.auth import router as auth_router
from app.api.settings import router as settings_router
from app.api.chat_history import router as chat_history_router
from app.config.settings import get_settings


@pytest.fixture(name="db_session")
def session_fixture():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(name="client")
def client_fixture(db_session):
    settings = get_settings()
    original_enable_auth = settings.enable_auth
    settings.enable_auth = True

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    test_app = FastAPI()
    test_app.include_router(auth_router, prefix="/api/v1")
    test_app.include_router(settings_router, prefix="/api/v1")
    test_app.include_router(chat_history_router, prefix="/api/v1")
    test_app.dependency_overrides[get_db] = override_get_db

    with TestClient(test_app, raise_server_exceptions=False) as c:
        yield c
    test_app.dependency_overrides.clear()
    settings.enable_auth = original_enable_auth


class TestSettingsEndpoints:
    def test_get_settings(self, client, db_session):
        create_user(db=db_session, email="user@test.com", name="User", password="Pass1234!", role=Role.EMPLOYEE)
        login_resp = client.post("/api/v1/auth/login", json={"email": "user@test.com", "password": "Pass1234!"})
        token = login_resp.json()["access_token"]
        response = client.get("/api/v1/settings", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        data = response.json()
        assert "llm_provider" in data
        assert "embedding_model" in data

    def test_update_settings_admin(self, client, db_session):
        create_user(db=db_session, email="admin@test.com", name="Admin", password="Pass1234!", role=Role.ADMIN)
        login_resp = client.post("/api/v1/auth/login", json={"email": "admin@test.com", "password": "Pass1234!"})
        token = login_resp.json()["access_token"]
        response = client.put(
            "/api/v1/settings",
            json={"llm_provider": "anthropic", "temperature": 0.5},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["llm_provider"] == "anthropic"

    def test_update_settings_requires_admin(self, client, db_session):
        create_user(db=db_session, email="user@test.com", name="User", password="Pass1234!", role=Role.EMPLOYEE)
        login_resp = client.post("/api/v1/auth/login", json={"email": "user@test.com", "password": "Pass1234!"})
        token = login_resp.json()["access_token"]
        response = client.put(
            "/api/v1/settings",
            json={"llm_provider": "anthropic"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403


class TestChatHistoryEndpoints:
    def test_create_session(self, client, db_session):
        create_user(db=db_session, email="user@test.com", name="User", password="Pass1234!", role=Role.EMPLOYEE)
        login_resp = client.post("/api/v1/auth/login", json={"email": "user@test.com", "password": "Pass1234!"})
        token = login_resp.json()["access_token"]
        response = client.post(
            "/api/v1/chat/sessions",
            json={"title": "Test Chat"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Test Chat"
        assert "id" in data

    def test_list_sessions(self, client, db_session):
        create_user(db=db_session, email="user@test.com", name="User", password="Pass1234!", role=Role.EMPLOYEE)
        login_resp = client.post("/api/v1/auth/login", json={"email": "user@test.com", "password": "Pass1234!"})
        token = login_resp.json()["access_token"]
        client.post(
            "/api/v1/chat/sessions",
            json={"title": "Chat 1"},
            headers={"Authorization": f"Bearer {token}"},
        )
        response = client.get("/api/v1/chat/sessions", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["total_count"] >= 1

    def test_add_message(self, client, db_session):
        create_user(db=db_session, email="user@test.com", name="User", password="Pass1234!", role=Role.EMPLOYEE)
        login_resp = client.post("/api/v1/auth/login", json={"email": "user@test.com", "password": "Pass1234!"})
        token = login_resp.json()["access_token"]
        session_resp = client.post(
            "/api/v1/chat/sessions",
            json={"title": "Test"},
            headers={"Authorization": f"Bearer {token}"},
        )
        session_id = session_resp.json()["id"]
        response = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            json={"role": "user", "content": "Hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 201
        assert response.json()["content"] == "Hello"

    def test_delete_session(self, client, db_session):
        create_user(db=db_session, email="user@test.com", name="User", password="Pass1234!", role=Role.EMPLOYEE)
        login_resp = client.post("/api/v1/auth/login", json={"email": "user@test.com", "password": "Pass1234!"})
        token = login_resp.json()["access_token"]
        session_resp = client.post(
            "/api/v1/chat/sessions",
            json={"title": "To Delete"},
            headers={"Authorization": f"Bearer {token}"},
        )
        session_id = session_resp.json()["id"]
        response = client.delete(
            f"/api/v1/chat/sessions/{session_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200

    def test_pin_session(self, client, db_session):
        create_user(db=db_session, email="user@test.com", name="User", password="Pass1234!", role=Role.EMPLOYEE)
        login_resp = client.post("/api/v1/auth/login", json={"email": "user@test.com", "password": "Pass1234!"})
        token = login_resp.json()["access_token"]
        session_resp = client.post(
            "/api/v1/chat/sessions",
            json={"title": "Pinnable"},
            headers={"Authorization": f"Bearer {token}"},
        )
        session_id = session_resp.json()["id"]
        response = client.put(
            f"/api/v1/chat/sessions/{session_id}",
            json={"is_pinned": True},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["is_pinned"] is True
