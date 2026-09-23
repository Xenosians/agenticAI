from scripts.jira_specialist_model_eval import (
    arguments_semantically_match,
)


def test_eval_accepts_bounded_optional_limit_default():
    assert arguments_semantically_match(
        tool_name="ticket_search",
        expected={
            "project_key": "KAN",
        },
        observed={
            "project_key": "KAN",
            "limit": 25,
        },
    )


def test_eval_rejects_invented_semantic_argument():
    assert not arguments_semantically_match(
        tool_name="ticket_search",
        expected={
            "project_key": "KAN",
        },
        observed={
            "project_key": "KAN",
            "status": "Done",
        },
    )


def test_eval_rejects_wrong_grounded_identifier():
    assert not arguments_semantically_match(
        tool_name="ticket_get",
        expected={
            "ticket_key": "KAN-3",
        },
        observed={
            "ticket_key": "KAN-4",
        },
    )
