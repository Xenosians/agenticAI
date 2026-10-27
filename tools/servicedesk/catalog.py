"""Jira Service Management: requests, SLAs and queues."""
from tools.servicedesk.policy import evaluate_service_request_create_policy
from tools.servicedesk.presentation import (
    build_queue_list_card,
    build_sla_status_card,
    format_queue_list_result,
    format_service_request_approval,
    format_service_request_result,
    format_sla_status_result,
)

SERVICE_DESK_PARAMETER = {
    "type": "str",
    "description": (
        "Exact service desk project key or name supplied by the user, such as SUP. "
        "Never invent or guess a service desk."
    ),
}

SERVICE_DESK_TOOLS = {
    "service_request_create": {
        "description": (
            "Raise one customer request in a Jira Service Management service desk using one named "
            "request type, such as Get IT help. Not for plain Jira tickets or issues. Preserve an "
            "exact user-supplied summary; otherwise compose it from the requested outcome. Never "
            "fills custom fields; request types that require them are refused."
        ),
        "risk": "medium",
        "requires_approval": True,
        "policy_resolver": evaluate_service_request_create_policy,
        "trusted_policy_arguments": [
            "expected_service_desk_id",
            "expected_request_type_id",
        ],
        "required_arguments": ["service_desk", "request_type", "summary"],
        "grounded_arguments": ["service_desk", "request_type", "description"],
        "derived_arguments": ["summary"],
        "parameters": {
            "service_desk": SERVICE_DESK_PARAMETER,
            "request_type": {
                "type": "str",
                "description": "Exact request type name supplied by the user, such as Get IT help.",
            },
            "summary": {
                "type": "str",
                "description": (
                    "Preserve an exact user-supplied summary. Otherwise compose a concise summary "
                    "from the requested outcome. Do not invent facts, people, dates or identifiers."
                ),
            },
            "description": {
                "type": "str",
                "description": "Optional exact description supplied by the user. Do not summarize or invent it.",
            },
        },
        "result_formatter": format_service_request_result,
        "approval_formatter": format_service_request_approval,
    },
    "sla_status": {
        "description": (
            "Read only: SLA status of one service desk request key, such as time to first response "
            "and time to resolution: ongoing, paused, breached or completed, with goal, elapsed and "
            "remaining time."
        ),
        "risk": "read",
        "requires_approval": False,
        "grounded_arguments": ["request_key"],
        "parameters": {
            "request_key": {
                "normalize":
                    "uppercase",

                "type": "str",
                "description": "Exact service desk request or issue key supplied by the user, such as SUP-12.",
            },
        },
        "condition_fields": ["ok", "count"],
        "result_formatter": format_sla_status_result,
        "presentation_builder": build_sla_status_card,
    },
    "queue_list": {
        "description": (
            "Read only: list the queues of one Jira Service Management service desk, named by project "
            "key or name, with the number of issues in each queue. Does not list or change the issues."
        ),
        "risk": "read",
        "requires_approval": False,
        "grounded_arguments": ["service_desk"],
        "parameters": {
            "service_desk": SERVICE_DESK_PARAMETER,
        },
        "condition_fields": ["ok", "count"],
        "result_formatter": format_queue_list_result,
        "presentation_builder": build_queue_list_card,
    },
}
