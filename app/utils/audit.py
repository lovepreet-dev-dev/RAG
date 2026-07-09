"""Audit logging utilities.

Provides helpers for recording security-critical actions in the audit_logs table.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.models.user import AuditLog


def log_audit(
    db: Session,
    *,
    user_id: str | None = None,
    action: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    details: str | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> AuditLog:
    """Write a single audit log entry.

    Args:
        db: Active SQLAlchemy session.
        user_id: ID of the user performing the action.
        action: Action identifier (e.g. ``login``, ``create_user``).
        resource_type: Type of resource affected (e.g. ``user``, ``document``).
        resource_id: ID of the affected resource.
        details: Free-text description.
        ip_address: Client IP address.
        user_agent: Client User-Agent string.

    Returns:
        The created ``AuditLog`` instance.
    """
    entry = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.add(entry)
    db.commit()
    return entry
