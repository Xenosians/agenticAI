import asyncio
import json
from pathlib import Path

import pytest

from config import Settings
from scripts.jira_specialist_model_eval import (
    CASES, build_messages, case_prompt, evaluation_options, evaluation_settings, load_cases, main,
    semantic_context,
)
from subagents.core.definitions.loader import load_agent_definition


def test_prompt_experiment_does_not_change_serving_profile():
    settings = Settings(_env_file=None, model_profiles={
        "candidate": {"backend": "hf-causal", "worker_prompt_profile": "compact"},
    })
    candidate = evaluation_settings(settings, model_key="candidate", prompt_profile="compact_grounded")
    assert candidate.require_model_profile("candidate").worker_prompt_profile == "compact_grounded"
    assert settings.require_model_profile("candidate").worker_prompt_profile == "compact"


def test_invalid_experimental_profile_is_rejected():
    settings = Settings(_env_file=None, model_profiles={"candidate": {"backend": "hf-causal"}})
    with pytest.raises(ValueError):
        evaluation_settings(settings, model_key="candidate", prompt_profile="invented")


def test_cli_accepts_explicit_experiment_and_report_path():
    args = evaluation_options(["--model-key", "candidate", "--prompt-profile", "compact_grounded",
                               "--report", "comparison.json"])
    assert args.model_key == "candidate"
    assert args.prompt_profile == "compact_grounded"
    assert args.report == Path("comparison.json")
    assert args.cases is None
    assert evaluation_options(["--cases", "holdout.json"]).cases == Path("holdout.json")


def write_case_file(tmp_path, case_list, **fields):
    path = tmp_path / "cases.json"
    payload = {"schema": "jira-evaluation-cases.v1", "case_set": "holdout", "cases": case_list}
    path.write_text(json.dumps({**payload, **fields}), encoding="utf-8")
    return path


def test_built_in_and_file_case_sets_carry_their_identity(tmp_path):
    built_in, built_in_digest, built_in_cases = load_cases(None)
    assert (built_in, built_in_cases) == ("development", CASES)
    case_set, digest, cases = load_cases(write_case_file(tmp_path, CASES[:2]))
    assert (case_set, cases) == ("holdout", CASES[:2])
    assert len(digest) == len(built_in_digest) == 64 and digest != built_in_digest


def renamed(case, name):
    return {**case, "name": name}


@pytest.mark.parametrize("fields", [
    {"schema": "evaluation-case.v1"},
    {"case_set": ""},
    {"cases": []},
    {"cases": [CASES[0], CASES[0]]},
    {"cases": [CASES[0], renamed(CASES[0], "same request")]},
    {"cases": [{key: value for key, value in CASES[0].items() if key != "expected_tool"}]},
    {"cases": [{**CASES[0], "allowed_tools": []}]},
    {"cases": [{**CASES[3], "expected_tool": "ticket_history"}]},
    {"cases": [{**CASES[3], "expected_arguments": "KAN-3"}]},
])
def test_malformed_case_files_are_rejected(tmp_path, fields):
    with pytest.raises(ValueError):
        load_cases(write_case_file(tmp_path, CASES[:1], **fields))


@pytest.mark.parametrize("payload", ["[]", "\"cases\"", "{not json"])
def test_case_file_must_be_a_json_object(tmp_path, payload):
    path = tmp_path / "cases.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(path)


def test_malformed_case_file_stops_before_settings_or_report(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    def unexpected_settings():
        raise AssertionError("No settings/model access for a malformed case file")
    monkeypatch.setattr("scripts.jira_specialist_model_eval.Settings", unexpected_settings)
    cases = write_case_file(tmp_path, [])
    assert asyncio.run(main(["--cases", str(cases), "--report", str(report)])) == 2
    assert not report.exists()


def _case_for(tool, **changes):
    return {**next(case for case in CASES if case["expected_tool"] == tool), **changes}


@pytest.mark.parametrize("case", [
    {key: value for key, value in CASES[0].items() if key != "allowed_arguments"},
    {**CASES[0], "allowed_tools": ["fixture_unknown_tool"], "expected_tool": "fixture_unknown_tool"},
    {**CASES[0], "allowed_tools": ["account_status"], "expected_tool": "account_status"},
    _case_for("ticket_get", expected_arguments={}),
    _case_for("ticket_get", expected_arguments={"ticket_key": "KAN-3", "fixture_unknown_field": "value"}),
    _case_for("jira_project_get", expected_arguments={"project_id_or_key": 7}),
    _case_for("jira_project_list", expected_arguments={"limit": True}),
    _case_for("ticket_update", expected_arguments={"ticket_key": "KAN-3", "add_labels": [7]}),
    _case_for("jira_project_create", expected_arguments={
        "project_key": "KAN", "project_name": "Fixture project", "template": "fixture_unknown_alias",
    }),
    _case_for("jira_project_create", allowed_arguments={"template": ["fixture_unknown_alias"]}),
    _case_for("ticket_get", allowed_arguments={"ticket_key": "KAN-3"}),
    _case_for("ticket_get", allowed_arguments={"ticket_key": []}),
    _case_for("ticket_get", allowed_arguments={"ticket_key": [""]}),
    _case_for("ticket_get", allowed_arguments={"ticket_key": [" KAN-3"]}),
    _case_for("ticket_get", allowed_arguments={"ticket_key": ["KAN-3", "KAN-3"]}),
    _case_for("ticket_get", allowed_arguments={"fixture_unknown_field": ["KAN-3"]}),
    {**CASES[0], "exact_arguments": "true"},
])
def test_invalid_case_contract_stops_before_settings_or_report(tmp_path, monkeypatch, case):
    report = tmp_path / "report.json"
    settings_calls = []

    def unexpected_settings():
        settings_calls.append(True)
        raise AssertionError("Case admission must precede deployment settings or models")

    monkeypatch.setattr("scripts.jira_specialist_model_eval.Settings", unexpected_settings)
    path = write_case_file(tmp_path, [case])
    assert asyncio.run(main(["--cases", str(path), "--report", str(report)])) == 2
    assert settings_calls == []
    assert not report.exists()


@pytest.mark.parametrize("cases", [1, "invalid", {"case": CASES[0]}])
def test_case_collection_must_be_an_explicit_list(tmp_path, cases):
    with pytest.raises(ValueError, match="cases"):
        load_cases(write_case_file(tmp_path, cases))


def test_built_in_and_held_out_cases_match_current_executable_capabilities():
    root = Path(__file__).resolve().parents[3]
    _, _, built_in = load_cases(None)
    _, _, held_out = load_cases(root / "learning/evaluation/evals/jira-specialist-holdout.json")
    assert len(built_in) == 33
    assert len(held_out) == 40
    assert {case["expected_tool"] for case in held_out} == {case["expected_tool"] for case in built_in}
    # The search set also loads against the executable contract, and every
    # expected search word or filter is the request's own wording, so the
    # gateway's quoting rule accepts each expected proposal.
    from subagents.core.tooling.gateway import validate_quoted_arguments
    from tools.registry import get_tool

    _, _, searches = load_cases(root / "learning/evaluation/evals/jira-search-holdout.json")
    assert {case["expected_tool"] for case in searches} == {"ticket_search"}
    quoted = get_tool("ticket_search")["quoted_arguments"]
    for case in searches:
        assert validate_quoted_arguments(case["user"], case["expected_arguments"], quoted) == (True, None), case["name"]


def test_existing_report_is_preserved_before_settings_or_inference(tmp_path, monkeypatch):
    report = tmp_path / "existing.json"
    report.write_text("historical evidence", encoding="utf-8")
    def unexpected_settings():
        raise AssertionError("No settings/model access after an existing report is found")
    monkeypatch.setattr("scripts.jira_specialist_model_eval.Settings", unexpected_settings)
    assert asyncio.run(main(["--report", str(report)])) == 2
    assert report.read_text(encoding="utf-8") == "historical evidence"


def test_evaluation_discloses_only_task_capabilities_and_keeps_original_request_last():
    root = Path(__file__).resolve().parents[3]
    agent = load_agent_definition(root / "subagents/agents/jira-specialist.md")
    case = {"user": "List Jira projects", "allowed_tools": ["jira_project_list"], "allowed_arguments": {}}
    catalog, prompt = case_prompt(agent, case, "compact_grounded")
    assert [item["name"] for item in catalog] == ["jira_project_list"]
    assert 'ticket_key' not in prompt
    assert 'KAN-3' not in prompt
    messages = build_messages(system_prompt=prompt, case=case, capability_catalog=catalog)
    assert messages[-1] == {"role": "user", "content": case["user"]}
    assert "expected_arguments" not in "\n".join(item["content"] for item in messages)


def test_evaluation_binds_only_grounded_targets_like_the_router():
    # The router drops derived proposal fields (a ticket summary) from the
    # semantic contract, and the training corpus does the same. Evaluation must
    # not hand the model a value that production makes it compose itself.
    from mcp_server import register_capability_signatures
    from subagents.core.orchestration.intent_contract import parse_semantic_intent
    from tools.capability_contract import build_capability_contract
    from tools.registry import get_tool

    register_capability_signatures()
    root = Path(__file__).resolve().parents[3]
    agent = load_agent_definition(root / "subagents/agents/jira-specialist.md")
    _, _, holdout = load_cases(root / "learning/evaluation/evals/jira-specialist-holdout.json")
    for case in [*CASES, *holdout]:
        catalog, prompt = case_prompt(agent, case, "compact_grounded")
        context = semantic_context(case, catalog)
        tool = case["allowed_tools"][0]
        routed = parse_semantic_intent({
            "summary": "evaluation case", "max_tool_calls": 1, "clarification_required": False,
            "effect": build_capability_contract(tool, get_tool(tool)).effect,
            "allowed_tools": case["allowed_tools"], "allowed_arguments": case["allowed_arguments"],
        }, agent=agent, user_request=case["user"])
        assert routed is not None
        assert set(context["allowed_arguments"]) == set(routed.allowed_arguments), case["name"]
        messages = build_messages(system_prompt=prompt, case=case, capability_catalog=catalog)
        assert messages[-1] == {"role": "user", "content": case["user"]}
    create = next(item for item in CASES if item["name"] == "issue create")
    catalog, prompt = case_prompt(agent, create, "compact_grounded")
    assert semantic_context(create, catalog)["allowed_arguments"] == {"project_key": ["KAN"]}
    messages = build_messages(system_prompt=prompt, case=create, capability_catalog=catalog)
    assert "VPN login fails after reboot" not in messages[1]["content"]
    assert create["expected_arguments"]["summary"] == "VPN login fails after reboot"
