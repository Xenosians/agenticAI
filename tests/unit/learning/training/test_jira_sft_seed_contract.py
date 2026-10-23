from learning.training.jira_sft import (
    JIRA_HELD_OUT_EVAL_REQUESTS,
    _build_records,
)

from subagents.core.tooling.parser import (
    parse_tool_calls,
)


REQUIRED_TOOLS = {
    "jira_project_list",
    "jira_project_get",
    "jira_project_create",
    "jira_project_update",
    "jira_project_archive",
    "jira_project_delete",
    "ticket_get",
    "ticket_search",
    "ticket_history",
    "ticket_comments",
    "ticket_add_comment",
    "ticket_create",
    "ticket_assign",
    "ticket_transition",
    "ticket_update",
    "ticket_link",
    "ticket_links",
    "ticket_assigned",
    "ticket_worklog_add",
}


def test_jira_sft_seed_covers_every_current_jira_capability():
    records = _build_records()

    observed = {
        record.tool_name
        for record in records
    }

    assert observed == REQUIRED_TOOLS
    assert len(records) >= 180


def test_jira_sft_seed_has_train_and_validation_for_every_tool():
    records = _build_records()

    for tool_name in REQUIRED_TOOLS:
        splits = {
            record.split
            for record in records
            if record.tool_name == tool_name
        }

        assert splits == {
            "train",
            "validation",
        }


def test_jira_sft_targets_are_exact_single_tool_calls():
    for record in _build_records():
        calls = parse_tool_calls(
            record.target_response
        )

        assert len(calls) == 1
        assert calls[0]["name"] == record.tool_name

        allowed = (
            record.semantic_context[
                "allowed_arguments"
            ]
        )

        for (
            key,
            value,
        ) in calls[0]["arguments"].items():
            assert allowed[key] == (
                value
                if isinstance(value, list)
                else [value]
            )


def test_jira_sft_seed_excludes_exact_held_out_eval_requests():
    requests = {
        record.user_request
        for record in _build_records()
    }

    assert requests.isdisjoint(
        JIRA_HELD_OUT_EVAL_REQUESTS
    )


def test_unfiltered_project_supervision_covers_both_splits_with_empty_arguments():
    records = [
        record for record in _build_records()
        if record.tool_name == "jira_project_list"
        and "empty-arguments" in record.tags
    ]

    assert {record.split for record in records} == {"train", "validation"}
    for record in records:
        assert parse_tool_calls(record.target_response)[0]["arguments"] == {}
        assert record.semantic_context["allowed_arguments"] == {}


def test_every_model_gate_case_is_explicitly_held_out_from_both_sft_splits():
    from scripts.jira_specialist_model_eval import CASES

    gate_requests = {case["user"] for case in CASES}
    assert gate_requests <= JIRA_HELD_OUT_EVAL_REQUESTS
    assert gate_requests.isdisjoint({record.user_request for record in _build_records()})
