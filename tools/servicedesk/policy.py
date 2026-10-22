"""Trusted policy for raising a service desk request."""
from __future__ import annotations

from typing import Any

from config import Settings
from services.servicedesk import NOT_CONFIGURED, ServiceDeskError, build_service_desk_service


def _decision(status: str, error: str) -> dict[str, Any]:
    return {
        "ok": False,
        "status": status,
        "risk": "medium",
        "requires_approval": True,
        "error": error,
    }


def evaluate_service_request_create_policy(
    service_desk: str,
    request_type: str,
    summary: str,
    description: str | None = None,
) -> dict[str, Any]:
    """
    Read-only preparation before approval.

    Resolves the service desk and request type, refuses request types whose
    required fields this tool cannot fill, and freezes both provider IDs so the
    approved request cannot land somewhere else.
    """

    arguments = {
        name: value
        for name, value in {
            "service_desk": service_desk,
            "request_type": request_type,
            "summary": summary,
            "description": description,
        }.items()
        if value is not None
    }

    service = None
    try:
        service = build_service_desk_service(Settings())
        if service is None:
            return _decision("error", NOT_CONFIGURED)

        frozen = service.prepare_request(**arguments)
        return {
            "ok": True,
            "status": "ready",
            "risk": "medium",
            "requires_approval": True,
            "execution_arguments": {**arguments, **frozen},
            "error": None,
        }
    except ServiceDeskError as exc:
        return _decision(exc.status, str(exc))
    except Exception:
        return _decision("error", "Trusted service request policy could not initialize the service desk provider.")
    finally:
        if service is not None:
            service.close()
