"""Text and result cards for service desk tools."""
from __future__ import annotations

from typing import Any

from tools.presentation.result_cards import build_result_card, list_section, result_field

# Bound the text handed back to the model; the full result stays structured.
MAX_LISTED_ITEMS = 25


def _failure(result: dict[str, Any], subject: str) -> str:
    return f"The service desk {subject} did not succeed: {result.get('error') or 'unknown error'}"


def _joined(lines: list[str]) -> str:
    listed = lines[:MAX_LISTED_ITEMS]
    hidden = len(lines) - len(listed)
    return "; ".join(listed) + (f"; and {hidden} more" if hidden > 0 else "")


def _metric_line(metric: dict[str, Any]) -> str:
    state = str(metric.get("state") or "unknown").replace("_", " ")
    parts = [f"{metric.get('name')}: {state}"]
    if metric.get("breached"):
        parts.append("BREACHED")
    if metric.get("remaining") and state in {"ongoing", "paused"}:
        parts.append(f"{metric['remaining']} remaining")
    if metric.get("elapsed"):
        parts.append(f"{metric['elapsed']} elapsed")
    if metric.get("goal"):
        parts.append(f"goal {metric['goal']}")
    return ", ".join(parts)


def _queue_line(queue: dict[str, Any]) -> str:
    count = queue.get("issue_count")
    if not isinstance(count, int):
        return f"{queue.get('name')} (count not reported)"
    return f"{queue.get('name')} ({count} issue{'' if count == 1 else 's'})"


def _items(result: dict[str, Any], field: str) -> list[dict[str, Any]]:
    items = result.get(field)
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def format_sla_status_result(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return _failure(result, "SLA lookup")
    key = result.get("request_key") or "the request"
    metrics = _items(result, "metrics")
    if not metrics:
        return f"{key} has no SLA metrics."
    return f"SLA status of {key}: " + _joined([_metric_line(metric) for metric in metrics])


def format_queue_list_result(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return _failure(result, "queue lookup")
    desk = f"{result.get('service_desk_name')} ({result.get('service_desk')})"
    queues = _items(result, "queues")
    if not queues:
        return f"The {desk} service desk has no queues."
    return f"Queues of the {desk} service desk: " + _joined([_queue_line(queue) for queue in queues])


def format_service_request_result(result: dict[str, Any]) -> str:
    return result.get("message") or result.get("error") or "The service request finished."


def format_service_request_approval(arguments: dict[str, Any]) -> str:
    request_type = arguments.get("request_type") or "a"
    desk = arguments.get("service_desk") or "the"
    message = (
        f"Raising a {request_type} request in the {desk} service desk requires approval, "
        f"with the summary: {arguments.get('summary')}"
    )
    description = arguments.get("description")
    if isinstance(description, str) and description.strip():
        message += f", and the description: {description}"
    return message if message.endswith((".", "!", "?")) else message + "."


def build_sla_status_card(result: dict[str, Any]) -> dict[str, Any]:
    key = result.get("request_key")
    return build_result_card(
        kind="service-desk-sla",
        title=f"SLA status of {key}" if isinstance(key, str) and key else "SLA status",
        status=str(result.get("status", "unknown")),
        fields=[
            result_field("Metrics", result.get("count")),
            result_field("Breached", sum(1 for metric in _items(result, "metrics") if metric.get("breached"))),
        ],
        sections=[list_section(title="Metrics", items=[_metric_line(metric) for metric in _items(result, "metrics")])],
    )


def build_queue_list_card(result: dict[str, Any]) -> dict[str, Any]:
    name = result.get("service_desk_name")
    return build_result_card(
        kind="service-desk-queues",
        title=f"{name} queues" if isinstance(name, str) and name else "Service desk queues",
        status=str(result.get("status", "unknown")),
        fields=[
            result_field("Service desk", result.get("service_desk")),
            result_field("Queues", result.get("count")),
        ],
        sections=[list_section(title="Queues", items=[_queue_line(queue) for queue in _items(result, "queues")])],
    )
