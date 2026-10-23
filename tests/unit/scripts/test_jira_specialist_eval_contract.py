from scripts.jira_specialist_model_eval import (
    arguments_semantically_match,
)

import pytest


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


@pytest.mark.parametrize("borrowed", [
    {"ticket_key": "KAN-3"},
    {"query": "KAN"},
    {"project_key": "KAN"},
    {"status": "Done"},
])
def test_unfiltered_project_eval_rejects_invented_targets_and_filters(borrowed):
    assert not arguments_semantically_match(
        tool_name="jira_project_list", expected={}, observed=borrowed,
    )


def test_filtered_project_eval_still_requires_the_exact_requested_query():
    assert arguments_semantically_match(
        tool_name="jira_project_list", expected={"query": "OPS"}, observed={"query": "OPS"},
    )
    assert not arguments_semantically_match(
        tool_name="jira_project_list", expected={"query": "OPS"}, observed={},
    )


@pytest.mark.parametrize("observed", [
    {"limit": 25}, {"limit": None}, {"ticket_key": "KAN-3"}, {"query": "OPS"},
])
def test_exact_unfiltered_project_case_rejects_every_extra_argument(observed):
    assert not arguments_semantically_match(
        tool_name="jira_project_list", expected={}, observed=observed, exact=True,
    )


def test_unfiltered_held_out_cases_require_exact_empty_arguments():
    from scripts.jira_specialist_model_eval import CASES

    empty_cases = [case for case in CASES if case["expected_arguments"] == {}]
    assert len(empty_cases) == 3
    assert all(case["exact_arguments"] is True for case in empty_cases)
    assert arguments_semantically_match(
        tool_name="jira_project_list", expected={}, observed={}, exact=True,
    )
