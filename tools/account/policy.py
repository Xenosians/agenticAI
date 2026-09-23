from __future__ import annotations

from typing import Any

from config import Settings
from services.directory import build_directory_service
from services.directory.account_creation import prepare_account_create


def evaluate_account_create_policy(
    given_name: str,
    family_name: str,
    department: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    settings = Settings()
    directory = None

    try:
        directory = build_directory_service(settings)
        return prepare_account_create(
            directory,
            settings,
            given_name=given_name,
            family_name=family_name,
            department=department,
            role=role,
        )
    except Exception:
        return {
            "ok": False,
            "status": "error",
            "risk": "high",
            "requires_approval": True,
            "error": "Trusted account-create policy could not prepare the account identity.",
        }
