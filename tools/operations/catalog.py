"""Platform operations: containers, service health and user notifications."""
from typing import Any

from tools.operations.policy import NOTIFY_USER_TRUSTED_ARGUMENTS, evaluate_notify_user_policy


def _failure_text(result: dict[str, Any], what: str) -> str:
    return f"The {what} did not succeed: {result.get('error') or 'the provider returned no details'}"


def format_container_status_result(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return _failure_text(result, "container status read")
    data = result.get("data") or {}
    lines = []
    for item in data.get("containers", []):
        if not item.get("found"):
            lines.append(f"{item.get('name')}: not found on this Docker host")
            continue
        details = [item.get("state") or "unknown"]
        if item.get("health"):
            details.append(f"health {item['health']}")
        if item.get("uptime_seconds") is not None:
            details.append(f"up {item['uptime_seconds']}s")
        elif item.get("exit_code") is not None:
            details.append(f"exit code {item['exit_code']}")
        details.append(f"restarts {item.get('restart_count')}")
        details.append(f"image {item.get('image')}")
        lines.append(f"{item.get('name')}: " + ", ".join(str(value) for value in details))
    summary = (f"{data.get('count', 0)} platform container(s): {data.get('running', 0)} running, "
               f"{data.get('unhealthy', 0)} unhealthy, {data.get('not_found', 0)} not found.")
    return "\n".join([summary, *lines])


def format_container_logs_result(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return _failure_text(result, "container log read")
    data = result.get("data") or {}
    notes = []
    if data.get("truncated"):
        notes.append("older output was cut to the size bound")
    if data.get("redaction_applied"):
        notes.append("secret-looking values were redacted")
    header = (f"Last {data.get('line_count', 0)} log line(s) of container {data.get('container')}"
              + (f" ({'; '.join(notes)})" if notes else "") + ":")
    return header + "\n" + (data.get("logs") or "(no log output)")


def format_service_health_result(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return _failure_text(result, "service health check")
    data = result.get("data") or {}
    lines = [f"{data.get('up', 0)} of {data.get('count', 0)} platform service(s) up."]
    for item in data.get("services", []):
        if item.get("up"):
            lines.append(f"{item.get('name')}: up (HTTP {item.get('http_status')}, {item.get('latency_ms')} ms)")
        elif item.get("http_status") is not None:
            lines.append(f"{item.get('name')}: down (HTTP {item.get('http_status')}, {item.get('latency_ms')} ms)")
        else:
            lines.append(f"{item.get('name')}: down ({item.get('error')})")
    return "\n".join(lines)



def format_notify_user_approval(arguments: dict[str, Any]) -> str:
    subject = arguments.get("subject")
    return (
        f"Send one notification email to the registered application user {arguments.get('email')}"
        + (f' with the subject "{subject}"' if subject else "")
        + f' and this exact message: "{arguments.get("message")}". '
        "The backend queues it in its mail outbox; this approval does not confirm delivery."
    )


def format_notify_user_result(result: dict[str, Any]) -> str:
    if result.get("ok") and result.get("delivery_status") == "sent":
        return (f"The backend previously recorded mail-provider acceptance of the notification to {result.get('email')} "
                f"(outbox {result.get('outbox_id')}). Inbox delivery is not confirmed.")
    if result.get("ok") and result.get("delivery_status") == "processing":
        return (f"Notification to {result.get('email')} is being processed by the mail outbox "
                f"({result.get('outbox_id')}). Delivery is not confirmed; do not submit it again.")
    if result.get("ok") and result.get("queued"):
        return (f"Notification to {result.get('email')} queued for delivery (outbox {result.get('outbox_id')}). "
                "It is queued, not confirmed as delivered.")
    if result.get("status") == "outcome_unknown":
        return (f"The notification outcome for {result.get('email')} is unknown: {result.get('error')} "
                "Do not resend it before checking the outbox.")
    return f"The notification was not queued: {result.get('error') or 'the backend gave no reason'}"


OPERATIONS_TOOLS: dict = {
    "container_status": {
        "description": (
            "Read only. State, health, uptime, restart count and image of the platform's own runtime containers "
            "on the allowlist, or of one named container. Never starts, stops or restarts containers."
        ),
        "risk": "read",
        "requires_approval": False,
        "grounded_arguments": ["container"],
        "parameters": {
            "container": {
                "type": "str",
                "description": (
                    "Optional exact platform container name explicitly supplied by the user. "
                    "Omit it to read every allow-listed container."
                ),
            },
        },
        "condition_fields": ["ok", "status"],
        "result_formatter": format_container_status_result,
    },
    "container_logs": {
        "description": (
            "Read only. The last log lines (1-200, default 50) of one allow-listed platform runtime container, "
            "size-bounded, with common secrets redacted. Does not follow, search or clear logs."
        ),
        "risk": "read",
        "requires_approval": False,
        "grounded_arguments": ["container"],
        "parameters": {
            "container": {
                "type": "str",
                "description": "Exact platform container name explicitly supplied by the user.",
            },
            "lines": {
                "type": "int",
                "description": (
                    "Optional number of most recent log lines from 1 to 200. "
                    "Use the number the user asks for; omit it for the default of 50."
                ),
            },
        },
        "condition_fields": ["ok", "status"],
        "result_formatter": format_container_logs_result,
    },
    "service_health": {
        "description": (
            "Read only. Check the platform's own health endpoints (backend and AI service): up or down, HTTP "
            "status and latency per service. Does not restart services or forward credentials."
        ),
        "risk": "read",
        "requires_approval": False,
        "grounded_arguments": [],
        "parameters": {},
        "condition_fields": ["ok", "status"],
        "result_formatter": format_service_health_result,
    },
    "notify_user": {
        "description": (
            "Send one notification email with the user's exact message to a registered application user through "
            "the backend mail outbox (queued, not confirmed as delivered). Requires approval."
        ),
        "risk": "medium",
        "requires_approval": True,
        "policy_resolver": evaluate_notify_user_policy,
        "trusted_policy_arguments": NOTIFY_USER_TRUSTED_ARGUMENTS,
        "grounded_arguments": ["email", "message", "subject"],
        "parameters": {
            "email": {
                "type": "str",
                "description": "Exact email address of the registered application user, as supplied by the user.",
            },
            "message": {
                "type": "str",
                "description": "Exact notification text supplied by the user. Do not rewrite, extend or summarize it.",
            },
            "subject": {
                "type": "str",
                "description": "Optional exact subject line supplied by the user; omit it when none is given.",
            },
        },
        "condition_fields": ["ok", "status", "queued"],
        "approval_formatter": format_notify_user_approval,
        "result_formatter": format_notify_user_result,
    },
}
