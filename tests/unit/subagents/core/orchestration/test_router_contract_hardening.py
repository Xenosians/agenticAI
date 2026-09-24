from subagents.core.orchestration.intent_contract import (
    _normalize_argument_map,
)


def test_null_optional_argument_is_omitted_fail_closed():
    assert _normalize_argument_map(
        {
            "summary": ["Alice Smith onboarding"],
            "ticket_type": None,
        },
        field_name="intent.allowed_arguments",
    ) == {
        "summary": ["Alice Smith onboarding"],
    }
