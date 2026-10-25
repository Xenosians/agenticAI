import json
import re
from pathlib import Path

from learning.training.jira_sft import (
    JIRA_HELD_OUT_EVAL_REQUESTS,
    _build_records,
)

from subagents.core.tooling.parser import (
    parse_tool_calls,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
HOLDOUT_CASES = PROJECT_ROOT / "learning/evaluation/evals/jira-specialist-holdout.json"


def _grounded_arguments():
    from mcp_server import register_capability_signatures
    from subagents.core.definitions.loader import load_agent_definition
    from subagents.core.tooling.capabilities import build_agent_capability_catalog

    register_capability_signatures()
    agent = load_agent_definition(PROJECT_ROOT / "subagents/agents/jira-specialist.md")
    return {
        item["name"]: set(item["grounded_arguments"])
        for item in build_agent_capability_catalog(agent, include_arguments=True)
    }


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


def _arguments(record):
    return parse_tool_calls(record.target_response)[0]["arguments"]


def _split_counts(records):
    return {split: sum(record.split == split for record in records) for split in ("train", "validation")}


def test_label_lists_are_taught_with_varied_lengths_and_companion_fields():
    # A label list must be closed before the next field. With only a handful of
    # list targets the model emitted unterminated lists on combined updates.
    updates = [
        record for record in _build_records()
        if record.tool_name == "ticket_update"
        and any(isinstance(value, list) for value in _arguments(record).values())
    ]
    counts = _split_counts(updates)
    assert counts["train"] >= 16 and counts["validation"] >= 4, counts

    arguments = [_arguments(record) for record in updates]
    assert {1, 2, 3} <= {len(item["add_labels"]) for item in arguments if "add_labels" in item}
    assert {1, 2} <= {len(item["remove_labels"]) for item in arguments if "remove_labels" in item}
    companions = {frozenset(item) - {"ticket_key"} for item in arguments}
    assert {
        frozenset({"add_labels", "remove_labels"}),
        frozenset({"add_labels", "priority"}),
        frozenset({"remove_labels", "priority"}),
        frozenset({"remove_labels", "due_date"}),
        frozenset({"add_labels", "due_date", "priority"}),
    } <= companions


def test_free_text_search_is_taught_alone_and_with_every_filter():
    # Search words are not router-bound, so the model must copy them even when a
    # project is bound beside them; it dropped them while such targets were rare.
    searches = [
        record for record in _build_records()
        if record.tool_name == "ticket_search" and "text" in _arguments(record)
    ]
    counts = _split_counts(searches)
    assert counts["train"] >= 16 and counts["validation"] >= 4, counts

    arguments = [_arguments(record) for record in searches]
    assert {
        frozenset(),
        frozenset({"project_key"}),
        frozenset({"status"}),
        frozenset({"priority"}),
        frozenset({"project_key", "status"}),
        frozenset({"project_key", "priority"}),
    } <= {frozenset(item) - {"text"} for item in arguments}
    assert any(len(item["text"].split()) >= 3 for item in arguments)
    assert any(any(character.isdigit() for character in item["text"]) for item in arguments)


def test_text_search_without_filters_is_the_default_shape():
    # A status or priority appears only when the user states one. When filtered
    # text searches outnumbered plain ones, a candidate invented a status for a
    # plain project-and-text search.
    searches = [
        _arguments(record) for record in _build_records()
        if record.tool_name == "ticket_search" and "text" in _arguments(record)
    ]
    filtered = sum(bool({"status", "priority"} & set(item)) for item in searches)
    assert len(searches) - filtered > filtered, {"plain": len(searches) - filtered, "filtered": filtered}


def test_an_empty_routed_context_mostly_teaches_empty_arguments():
    # The runtime binds only grounded values, so unfiltered project listing and a
    # search without a project both reach the model with no bound values; only
    # the tool tells them apart. At an even 10:10 split a candidate put a text
    # argument on project listing, so keep empty targets at least 2:1.
    grounded = _grounded_arguments()
    routed_empty = [
        record for record in _build_records()
        if not set(record.semantic_context["allowed_arguments"]) & grounded[record.tool_name]
    ]
    emitting = sum(bool(_arguments(record)) for record in routed_empty)
    assert len(routed_empty) - emitting >= 2 * emitting, {
        "empty": len(routed_empty) - emitting, "emitting": emitting,
    }


def _values(arguments):
    for value in arguments.values():
        yield from (value if isinstance(value, list) else [value])


def _request_template(request, arguments):
    text = request
    for value in sorted({str(item) for item in _values(arguments)}, key=len, reverse=True):
        text = text.replace(value, "<v>")
    return re.sub(r"\s+", " ", text.strip().casefold())


def test_holdout_cases_share_no_request_template_or_value_with_training():
    # The held-out set scores generalization: a training record with the same
    # request shape, or the same ticket, label or search text, would turn it
    # into a replay check.
    cases = json.loads(HOLDOUT_CASES.read_text(encoding="utf-8"))["cases"]
    records = [(record.user_request, _arguments(record)) for record in _build_records()]
    training_templates = {_request_template(request, arguments) for request, arguments in records}
    training_values = {str(value).casefold() for _, arguments in records for value in _values(arguments)}
    closed_vocabulary = {
        str(value).casefold()
        for case in cases
        for name, value in case["expected_arguments"].items()
        if name in {"status", "priority", "ticket_type", "template", "relationship"}
    }
    for case in cases:
        assert _request_template(case["user"], case["expected_arguments"]) not in training_templates, case["name"]
        shared = {str(value).casefold() for value in _values(case["expected_arguments"])} & training_values
        assert shared <= closed_vocabulary, (case["name"], shared)
