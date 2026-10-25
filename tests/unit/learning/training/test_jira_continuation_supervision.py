"""The account -> Jira continuation step is taught and scored with the live prompt.

At this step the runtime adds a verified-account system message and the model must
compose the required ticket summary from it. A model never trained or scored on
that message omitted the summary in a live run, so the corpus and the evaluation
must render exactly the messages the live worker sends.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent.onboarding import VerifiedAccountEvidence
from config import ModelProfileSettings, Settings
from learning.training import jira_sft
from subagents.core.definitions.registry import AgentRegistry
from subagents.core.definitions.types import AgentTask, SemanticIntent
from subagents.core.orchestration.runtime import AgentRuntime
from subagents.core.tooling.parser import parse_tool_calls


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONTINUATION_CASES = PROJECT_ROOT / "learning/evaluation/evals/jira-continuation-holdout.json"
EVIDENCE_PREFIX = "VERIFIED WORKFLOW CONTENT:\n"


def _jira_agent():
    from mcp_server import register_capability_signatures

    register_capability_signatures()
    return jira_sft.load_agent_definition(PROJECT_ROOT / "subagents/agents/jira-specialist.md")


def _continuation_records():
    return [record for record in jira_sft._build_records() if record.verified_account is not None]


def _live_worker_messages(agent, user_request, project_key, account):
    """Capture the messages the production worker sends for one continuation."""
    registry = AgentRegistry()
    registry.register(agent)
    captured = {}

    class InferenceCapture:
        async def generate(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("capture stops before parsing or any tool")

    runtime = AgentRuntime(
        agent_registry=registry, inference=InferenceCapture(), tool_gateway=object(),
        model_profile_resolver=lambda _: ModelProfileSettings(
            backend="hf-causal", worker_prompt_profile="compact_grounded",
        ),
    )
    asyncio.run(runtime.run(AgentTask(
        task_id="continuation-capture", agent_name=agent.name, user_request=user_request,
        semantic_intent=SemanticIntent(
            summary="Record the requested onboarding", effect="mutation",
            allowed_tools=["ticket_create"], allowed_arguments={"project_key": [project_key]},
        ),
        context={"verified_account": VerifiedAccountEvidence(**account)},
    )))
    return captured["messages"]


def test_corpus_teaches_the_continuation_in_both_splits():
    records = _continuation_records()
    assert {record.split for record in records} == {"train", "validation"}
    assert sum(record.split == "train" for record in records) >= 8
    assert all(record.tool_name == "ticket_create" for record in records)


def test_continuation_targets_carry_the_bound_project_and_name_the_verified_account():
    for record in _continuation_records():
        arguments = parse_tool_calls(record.target_response)[0]["arguments"]
        assert record.semantic_context["allowed_arguments"]["project_key"] == [arguments["project_key"]]
        assert arguments["summary"].strip()
        account = record.verified_account
        assert account["user_id"] in arguments["summary"] or account["email"] in arguments["summary"]
        assert "ticket_type" not in arguments


def test_runtime_corpus_renders_the_live_continuation_messages(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, model_profiles={
        "jira-func": ModelProfileSettings(backend="hf-causal", worker_prompt_profile="compact"),
    })
    monkeypatch.setattr(jira_sft, "Settings", lambda: settings)
    directory = tmp_path / "corpus"
    jira_sft.build_jira_sft_corpus(
        project_root=PROJECT_ROOT, output_directory=directory,
        worker_prompt_profile="compact_grounded", prompt_layout="runtime_task",
    )
    _, train, validation, _ = jira_sft.verify_jira_sft_corpus(
        project_root=PROJECT_ROOT, corpus_directory=directory, worker_prompt_profile="compact_grounded",
    )
    agent = _jira_agent()
    records = [record for record in [*train, *validation] if record.verified_account is not None]
    assert records
    for record in records:
        live = _live_worker_messages(
            agent, record.user_request,
            record.semantic_context["allowed_arguments"]["project_key"][0], record.verified_account,
        )
        assert record.prompt_messages == live
        assert any(message["content"].startswith(EVIDENCE_PREFIX) for message in record.prompt_messages)


def test_runtime_corpus_rejects_a_tampered_verified_account(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, model_profiles={
        "jira-func": ModelProfileSettings(backend="hf-causal", worker_prompt_profile="compact"),
    })
    monkeypatch.setattr(jira_sft, "Settings", lambda: settings)
    directory = tmp_path / "corpus"
    jira_sft.build_jira_sft_corpus(
        project_root=PROJECT_ROOT, output_directory=directory,
        worker_prompt_profile="compact_grounded", prompt_layout="runtime_task",
    )
    for split in ("train", "validation"):
        path = directory / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            if row.get("verified_account"):
                row["verified_account"]["email"] = "someone.else@example.org"
                break
        else:
            continue
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest[f"{split}_sha256"] = jira_sft._sha256_file(path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        break
    with pytest.raises(ValueError, match="runtime-task messages changed"):
        jira_sft.verify_jira_sft_corpus(
            project_root=PROJECT_ROOT, corpus_directory=directory, worker_prompt_profile="compact_grounded",
        )


def test_evaluation_renders_the_live_continuation_messages():
    from scripts.jira_specialist_model_eval import build_messages, case_prompt, load_cases

    case_set, _, cases = load_cases(CONTINUATION_CASES)
    assert case_set == "continuation-holdout"
    agent = _jira_agent()
    for case in cases:
        assert case["expected_tool"] == "ticket_create"
        catalog, prompt = case_prompt(agent, case, "compact_grounded")
        messages = build_messages(system_prompt=prompt, case=case, capability_catalog=catalog)
        live = _live_worker_messages(
            agent, case["user"], case["allowed_arguments"]["project_key"][0], case["verified_account"],
        )
        assert messages == live, case["name"]


def test_continuation_cases_are_held_out_from_training():
    from scripts.jira_specialist_model_eval import load_cases

    _, _, cases = load_cases(CONTINUATION_CASES)
    training = jira_sft._build_records()
    requests = {record.user_request for record in training}
    accounts = {
        value.casefold()
        for record in training if record.verified_account is not None
        for value in record.verified_account.values()
    }
    projects = {
        str(parse_tool_calls(record.target_response)[0]["arguments"].get("project_key", "")).casefold()
        for record in training
    }
    for case in cases:
        assert case["user"] not in requests, case["name"]
        assert not {value.casefold() for value in case["verified_account"].values()} & accounts, case["name"]
        assert case["allowed_arguments"]["project_key"][0].casefold() not in projects, case["name"]


def test_a_continuation_case_scores_the_summary_against_the_verified_account():
    from scripts.jira_specialist_model_eval import continuation_proposal_passes, load_cases

    _, _, cases = load_cases(CONTINUATION_CASES)
    case = cases[0]
    account = case["verified_account"]
    project = case["allowed_arguments"]["project_key"][0]
    good = {"project_key": project, "summary": f"Onboarding follow-up for {account['user_id']}"}
    assert continuation_proposal_passes(case, "ticket_create", good)
    assert continuation_proposal_passes(case, "ticket_create", {**good, "summary": account["email"]})
    assert not continuation_proposal_passes(case, "ticket_create", {"project_key": project})
    assert not continuation_proposal_passes(case, "ticket_create", {**good, "summary": "Onboarding follow-up"})
    assert not continuation_proposal_passes(case, "ticket_create", {**good, "project_key": "OTHER"})
    assert not continuation_proposal_passes(case, "ticket_create", {**good, "ticket_type": "Task"})
    assert not continuation_proposal_passes(case, "ticket_get", good)
