"""Prompt versioning and management.

Supports multiple prompt versions with rollback capability.  Each prompt
template is stored with a version tag and can be activated/deactivated.
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
class PromptVersion:
    """A versioned prompt template."""

    name: str
    version: str
    template: str
    description: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptVersion:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PromptManager:
    """Manages versioned prompt templates with rollback support.

    Persists prompt versions to disk as JSON.  Supports:
    - Creating new prompt versions
    - Activating/deactivating versions
    - Rolling back to a previous version
    - Listing all versions of a prompt
    """

    def __init__(self, persist_directory: str | Path | None = None) -> None:
        self._persist_directory = Path(persist_directory) if persist_directory else None
        self._prompts: dict[str, list[PromptVersion]] = {}
        self._active: dict[str, str] = {}  # name -> version
        self._loaded = False

    def load(self) -> None:
        """Load persisted prompt versions from disk."""
        if self._persist_directory is None:
            return

        prompts_dir = self._persist_directory / "prompts"
        if not prompts_dir.exists():
            return

        for prompt_file in prompts_dir.glob("*.json"):
            try:
                data = json.loads(prompt_file.read_text())
                name = data.get("name", prompt_file.stem)
                versions = [PromptVersion.from_dict(v) for v in data.get("versions", [])]
                self._prompts[name] = versions
                active = data.get("active_version")
                if active:
                    self._active[name] = active
            except Exception as exc:
                logger.warning("prompt_load_failed", file=str(prompt_file), error=str(exc))

        self._loaded = True

    def save(self) -> None:
        """Persist all prompt versions to disk."""
        if self._persist_directory is None:
            return

        prompts_dir = self._persist_directory / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        for name, versions in self._prompts.items():
            data = {
                "name": name,
                "versions": [v.to_dict() for v in versions],
                "active_version": self._active.get(name),
            }
            path = prompts_dir / f"{name}.json"
            path.write_text(json.dumps(data, indent=2))

    def create_version(
        self,
        name: str,
        template: str,
        version: str | None = None,
        description: str = "",
        activate: bool = True,
    ) -> PromptVersion:
        """Create a new version of a prompt template.

        Args:
            name: Prompt name (e.g., 'rag_system', 'rag_user').
            template: The prompt template text.
            version: Version string (auto-generated if not provided).
            description: Description of this version.
            activate: Whether to activate this version immediately.

        Returns:
            The created PromptVersion.
        """
        if not self._loaded:
            self.load()

        existing = self._prompts.get(name, [])
        if version is None:
            version = f"v{len(existing) + 1}.0.0"

        prompt_version = PromptVersion(
            name=name,
            version=version,
            template=template,
            description=description,
            is_active=activate,
        )

        self._prompts.setdefault(name, []).append(prompt_version)

        if activate:
            self._active[name] = version

        self.save()

        logger.info(
            "prompt_version_created",
            name=name,
            version=version,
            activated=activate,
        )

        return prompt_version

    def get_active(self, name: str) -> str | None:
        """Get the active template for a prompt name.

        Args:
            name: Prompt name.

        Returns:
            Active template text, or None if not found.
        """
        if not self._loaded:
            self.load()

        active_version = self._active.get(name)
        if active_version is None:
            return None

        for v in self._prompts.get(name, []):
            if v.version == active_version and v.is_active:
                return v.template

        return None

    def rollback(self, name: str, version: str) -> bool:
        """Rollback to a previous prompt version.

        Args:
            name: Prompt name.
            version: Version to rollback to.

        Returns:
            True if rollback was successful.
        """
        if not self._loaded:
            self.load()

        for v in self._prompts.get(name, []):
            if v.version == version:
                # Deactivate current
                current_version = self._active.get(name)
                for pv in self._prompts.get(name, []):
                    if pv.version == current_version:
                        pv.is_active = False

                # Activate target
                v.is_active = True
                self._active[name] = version
                self.save()

                logger.info("prompt_rollback", name=name, to_version=version)
                return True

        logger.warning("prompt_rollback_failed", name=name, version=version)
        return False

    def list_versions(self, name: str) -> list[PromptVersion]:
        """List all versions of a prompt.

        Args:
            name: Prompt name.

        Returns:
            List of PromptVersion objects.
        """
        if not self._loaded:
            self.load()

        return self._prompts.get(name, [])

    def list_prompts(self) -> list[str]:
        """List all prompt names.

        Returns:
            List of prompt names.
        """
        if not self._loaded:
            self.load()

        return list(self._prompts.keys())


# Default prompt manager instance
_prompt_manager: PromptManager | None = None


def get_prompt_manager(persist_directory: str | Path | None = None) -> PromptManager:
    """Get or create the global prompt manager.

    Args:
        persist_directory: Directory for persisting prompt versions.

    Returns:
        PromptManager instance.
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager(persist_directory)
    return _prompt_manager
