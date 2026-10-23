import asyncio
from pathlib import Path

import pytest

from config import Settings
from scripts.jira_specialist_model_eval import (
    build_messages, case_prompt, evaluation_options, evaluation_settings, main,
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
    messages = build_messages(system_prompt=prompt, case=case)
    assert messages[-1] == {"role": "user", "content": case["user"]}
    assert "expected_arguments" not in "\n".join(item["content"] for item in messages)
