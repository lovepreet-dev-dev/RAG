"""Tests for authentication endpoints."""

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
    test_app.dependency_overrides[get_db] = override_get_db

    with TestClient(test_app, raise_server_exceptions=False) as c:
        yield c
    test_app.dependency_overrides.clear()
    settings.enable_auth = original_enable_auth


class TestAuthEndpoints:
    def test_login_success(self, client, db_session):
        create_user(db=db_session, email="test@test.com", name="Test", password="Pass1234!", role=Role.EMPLOYEE)
        response = client.post("/api/v1/auth/login", json={"email": "test@test.com", "password": "Pass1234!"})
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_login_invalid_credentials(self, client, db_session):
        create_user(db=db_session, email="test@test.com", name="Test", password="Pass1234!", role=Role.EMPLOYEE)
        response = client.post("/api/v1/auth/login", json={"email": "test@test.com", "password": "wrong"})
        assert response.status_code == 401

    def test_register_user(self, client):
        response = client.post("/api/v1/auth/register", json={
            "email": "new@test.com",
            "name": "New User",
            "password": "Pass1234!",
            "role": "employee",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "new@test.com"
        assert data["role"] == "employee"

    def test_get_me(self, client, db_session):
        create_user(db=db_session, email="test@test.com", name="Test", password="Pass1234!", role=Role.EMPLOYEE)
        login_resp = client.post("/api/v1/auth/login", json={"email": "test@test.com", "password": "Pass1234!"})
        token = login_resp.json()["access_token"]
        response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["email"] == "test@test.com"

    def test_list_users_requires_admin(self, client, db_session):
        create_user(db=db_session, email="emp@test.com", name="Emp", password="Pass1234!", role=Role.EMPLOYEE)
        login_resp = client.post("/api/v1/auth/login", json={"email": "emp@test.com", "password": "Pass1234!"})
        token = login_resp.json()["access_token"]
        response = client.get("/api/v1/auth/users", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 403

    def test_list_users_admin(self, client, db_session):
        create_user(db=db_session, email="admin@test.com", name="Admin", password="Pass1234!", role=Role.ADMIN)
        login_resp = client.post("/api/v1/auth/login", json={"email": "admin@test.com", "password": "Pass1234!"})
        token = login_resp.json()["access_token"]
        response = client.get("/api/v1/auth/users", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["total_count"] >= 1

    def test_refresh_token(self, client, db_session):
        create_user(db=db_session, email="test@test.com", name="Test", password="Pass1234!", role=Role.EMPLOYEE)
        login_resp = client.post("/api/v1/auth/login", json={"email": "test@test.com", "password": "Pass1234!"})
        refresh = login_resp.json()["refresh_token"]
        response = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_change_password(self, client, db_session):
        create_user(db=db_session, email="test@test.com", name="Test", password="Pass1234!", role=Role.EMPLOYEE)
        login_resp = client.post("/api/v1/auth/login", json={"email": "test@test.com", "password": "Pass1234!"})
        token = login_resp.json()["access_token"]
        response = client.post(
            "/api/v1/auth/change-password",
            json={"current_password": "Pass1234!", "new_password": "NewPass123!"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
