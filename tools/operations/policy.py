"""Trusted preparation for platform operations mutations; nothing here sends anything."""
from typing import Any

from config import Settings
from services.notifications import UserNotificationService, new_dedupe_key, validate_notification


NOTIFY_USER_TRUSTED_ARGUMENTS = ["dedupe_key"]


def evaluate_notify_user_policy(email: str, message: str, subject: str | None = None) -> dict[str, Any]:
    """Validate the exact request and freeze its dedupe key before approval."""
    def refused(status: str, error: str) -> dict[str, Any]:
        return {"ok": False, "status": status, "risk": "medium", "requires_approval": True, "error": error}

    invalid = validate_notification(email, message, subject)
    if invalid:
        return refused("denied", invalid)
    try:
        UserNotificationService(Settings()).validate_configuration()
    except RuntimeError as exc:
        return refused("error", str(exc))

    arguments: dict[str, Any] = {"email": email, "message": message}
    if subject is not None:
        arguments["subject"] = subject
    return {
        "ok": True,
        "status": "ready",
        "risk": "medium",
        "requires_approval": True,
        "execution_arguments": {**arguments, "dedupe_key": new_dedupe_key(email, message, subject)},
        "error": None,
    }
