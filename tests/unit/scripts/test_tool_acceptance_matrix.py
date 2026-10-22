import asyncio
import json
from dataclasses import replace

import pytest

from scripts.tool_acceptance_matrix import (
    ADAPTERS,
    build_report,
    evaluate_proposal,
    generate_observations,
    load_inventory,
    load_proposal_cases,
    select_cases,
    validate_case,
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    from config import Settings
    from services import git_repositories

    # These proposal fixtures name three logical repositories. Declare that
    # test deployment explicitly instead of reading aliases from a user's .env.
    root = tmp_path_factory.mktemp("proposal-repositories")
    aliases = {name: root / name for name in ("ai", "backend", "frontend")}
    for path in aliases.values():
        path.mkdir()
    settings = Settings(_env_file=None, git_repositories=aliases, git_default_repository="ai")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(git_repositories, "get_settings", lambda: settings)
        patch.setattr(git_repositories, "workspace_root", lambda: root)
        contracts, agents = load_inventory()
        yield contracts, agents, load_proposal_cases()


def eligible(corpus, tool):
    contracts, agents, cases = corpus
    return next(case for case in cases if case.expected_tool == tool
                and not case.training_overlap and not validate_case(case, contracts, agents))


def response(case, *, arguments=None):
    return json.dumps([{"name": case.expected_tool, "arguments": case.expected_arguments if arguments is None else arguments}])


def observation(case, passed=True):
    return {"adapter": "http", "case_id": case.case_id,
            "case_sha256": case.digest, "passed": passed, "decision_code": "fixture"}


def test_every_runtime_tool_has_a_valid_proposal_fixture(corpus):
    contracts, agents, cases = corpus
    report = build_report(contracts=contracts, agents=agents, cases=cases, references={})
    from tools.registry import TOOLS

    assert {row["name"] for row in report["tools"]} == set(TOOLS)
    assert report["summary"]["tools_missing_valid_proposal_cases"] == []
    assert report["summary"]["tools_without_specialist"] == []
    assert report["summary"]["invalid_cases"] == []
    assert report["summary"]["tools_missing_existing_held_out_proposal_cases"]
    assert report["summary"]["tools_missing_reviewed_proposal_cases"] == []
    assert report["summary"]["authored_unreviewed_cases"] == 0
    assert report["summary"]["all_tools_mvp_verified"] is False
    for row in report["tools"]:
        assert row["requiredness_source"] == "executable_signature"
        for adapter in ADAPTERS:
            assert row["adapters"][adapter]["proposal"] == "unverified"
            assert row["adapters"][adapter]["tool_execution"] == "unverified"
            assert row["adapters"][adapter]["durable_app_flow"] == "unverified"


@pytest.mark.parametrize("arguments,code", [
    ({"asset_id": "AST-999"}, "semantic_argument_not_allowed"),
    ({"asset_id": "AST-204", "credential": "never-disclose"}, "unknown_arguments"),
    ({}, "semantic_required_argument_missing"),
])
def test_proposal_validation_rejects_wrong_grounding_shape_and_missing_values(corpus, arguments, code):
    contracts, agents, _ = corpus
    case = eligible(corpus, "asset_get")
    result = evaluate_proposal(case, response(case, arguments=arguments), contracts=contracts, agents=agents)
    assert result == {"passed": False, "decision_code": code}


def test_intent_validation_rejects_fixture_targets_absent_from_user_request(corpus):
    contracts, agents, _ = corpus
    case = replace(eligible(corpus, "asset_get"), user="Read an asset but I have not supplied its identifier.")
    assert validate_case(case, contracts, agents) == "invalid_gold:semantic_clarification_required"


def test_multi_call_output_fails_without_executing_either_call(corpus):
    contracts, agents, _ = corpus
    case = eligible(corpus, "asset_get")
    raw = json.dumps(json.loads(response(case)) * 2)
    assert evaluate_proposal(case, raw, contracts=contracts, agents=agents)["decision_code"] == "invalid_call_count"


def test_collection_grounding_rejects_added_file(corpus):
    contracts, agents, _ = corpus
    case = eligible(corpus, "workspace_git_unstage_files")
    arguments = {**case.expected_arguments, "paths": [*case.expected_arguments["paths"], "secrets.env"]}
    result = evaluate_proposal(case, response(case, arguments=arguments), contracts=contracts, agents=agents)
    assert result["passed"] is False
    assert result["decision_code"] == "semantic_argument_not_allowed"


def test_authored_success_does_not_claim_review_execution_or_app_flow(corpus):
    contracts, agents, _ = corpus
    case = replace(
        eligible(corpus, "asset_get"),
        case_id="synthetic-authored-unreviewed-case",
        review_status="authored_unreviewed",
    )
    report = build_report(contracts=contracts, agents=agents, cases=[case], references={},
                          observations=[observation(case)])
    row = next(row for row in report["tools"] if row["name"] == case.expected_tool)
    state = row["adapters"]["http"]
    assert state["proposal"] == "passed_authored_unreviewed"
    assert state["tool_execution"] == state["durable_app_flow"] == "unverified"
    assert report["summary"]["all_tools_mvp_verified"] is False


def test_partial_evidence_and_training_overlap_cannot_be_credited_as_full(corpus):
    contracts, agents, _ = corpus
    first = eligible(corpus, "asset_get")
    second = replace(first, case_id="second independent case")
    report = build_report(contracts=contracts, agents=agents, cases=[first, second], references={},
                          observations=[observation(first)])
    row = next(row for row in report["tools"] if row["name"] == first.expected_tool)
    assert row["adapters"]["http"]["proposal"] == "partial"
    leaked = replace(first, training_overlap=True)
    report = build_report(contracts=contracts, agents=agents, cases=[leaked], references={},
                          observations=[observation(leaked)])
    row = next(row for row in report["tools"] if row["name"] == first.expected_tool)
    assert row["adapters"]["http"]["proposal"] == "missing_cases"


def test_duplicate_or_obsolete_observations_fail_closed(corpus):
    contracts, agents, _ = corpus
    case = eligible(corpus, "asset_get")
    for records in ([observation(case), observation(case)],
                    [{**observation(case), "case_sha256": "obsolete"}]):
        with pytest.raises(ValueError, match="unique current valid"):
            build_report(contracts=contracts, agents=agents, cases=[case], references={}, observations=records)


def test_case_selection_is_explicit_bounded_and_excludes_training_overlap(corpus):
    contracts, agents, _ = corpus
    case = eligible(corpus, "asset_get")
    assert select_cases([case], requested=[case.case_id], limit=1, contracts=contracts, agents=agents) == [case]
    with pytest.raises(ValueError):
        select_cases([replace(case, training_overlap=True)], requested=[case.case_id], limit=1,
                     contracts=contracts, agents=agents)
    with pytest.raises(ValueError):
        select_cases([case], requested=[case.case_id, case.case_id], limit=1, contracts=contracts, agents=agents)


def test_networking_generation_records_actual_local_stage_for_external_selection(corpus):
    contracts, agents, _ = corpus
    case = eligible(corpus, "network_system_info")

    class Inference:
        async def generate(self, **kwargs):
            assert kwargs["stage"].value == "NETWORKING"
            assert kwargs["model_key"] == agents[case.agent_name].model
            return response(case)

    results = asyncio.run(generate_observations(
        inference=Inference(), adapter="http", model_key="must-not-override-networking",
        cases=[case], contracts=contracts, agents=agents, max_tokens=128, timeout=1,
    ))
    assert results[0]["passed"] is True
    assert results[0]["adapter"] == "local"
    assert results[0]["requested_adapter"] == "http"


def test_opted_networking_generation_reports_selected_stage_and_never_claims_execution(corpus):
    from services.extensions.context import execution_profile

    contracts, agents, _ = corpus
    case = eligible(corpus, "network_system_info")

    class Inference:
        async def generate(self, **kwargs):
            assert kwargs["stage"].value == "NETWORKING_SELECTED"
            return response(case)

    token = execution_profile.set({"backend": "http", "model": "selected-model",
                                   "networking_inference": "selected_model"})
    try:
        results = asyncio.run(generate_observations(
            inference=Inference(), adapter="http", model_key="networking-default",
            cases=[case], contracts=contracts, agents=agents, max_tokens=128, timeout=1,
        ))
    finally:
        execution_profile.reset(token)
    assert results[0]["passed"] is True
    assert results[0]["adapter"] == "http"
    assert results[0]["stage"] == "NETWORKING_SELECTED"
    assert results[0]["networking_inference"] == "selected_model"
    assert results[0]["tool_executed"] is False
    report = build_report(contracts=contracts, agents=agents, cases=[case], references={}, observations=results)
    row = next(row for row in report["tools"] if row["name"] == case.expected_tool)
    assert row["adapters"]["http"]["effective_adapter"] == "http"
    assert row["adapters"]["http"]["observed_inference_stages"] == ["NETWORKING_SELECTED"]
    assert row["adapters"]["http"]["observed_networking_modes"] == ["selected_model"]
    assert row["adapters"]["http"]["tool_execution"] == "unverified"
    assert report["summary"]["all_tools_mvp_verified"] is False


def test_generation_errors_are_counted_without_leaking_error_content(corpus):
    contracts, agents, _ = corpus
    case = eligible(corpus, "asset_get")

    class Inference:
        async def generate(self, **kwargs):
            assert kwargs["model_key"] == "explicit-baseline"
            assert kwargs["stage"].value == "SPECIALIST_REASONING"
            raise RuntimeError("credential=must-never-appear-in-report")

    results = asyncio.run(generate_observations(
        inference=Inference(), adapter="local", model_key="explicit-baseline", cases=[case],
        contracts=contracts, agents=agents, max_tokens=128, timeout=1,
    ))
    assert results[0]["passed"] is False
    assert results[0]["error_type"] == "RuntimeError"
    assert "must-never-appear" not in json.dumps(results)


def test_external_generation_uses_real_runtime_filtered_catalog_and_mandatory_request(corpus):
    from services.extensions.context import execution_profile

    contracts, agents, _ = corpus
    case = eligible(corpus, "asset_get")

    class Inference:
        async def generate(self, **kwargs):
            messages = kwargs["messages"]
            assert messages[-1] == {"role": "user", "content": case.user}
            assert any(message["role"] == "system" and "Exact semantic target context" in message["content"]
                       for message in messages)
            assert '"name": "asset_get"' in messages[0]["content"]
            assert '"name": "asset_assign"' not in messages[0]["content"]
            return response(case)

    token = execution_profile.set({"backend": "http", "model": "synthetic-test-profile"})
    try:
        results = asyncio.run(generate_observations(
            inference=Inference(), adapter="http", model_key="unused-external-logical-key",
            cases=[case], contracts=contracts, agents=agents, max_tokens=128, timeout=1,
        ))
    finally:
        execution_profile.reset(token)
    assert results[0]["passed"] is True
    assert results[0]["runtime_semantic_guard_passed"] is True
    assert results[0]["runtime_outcome_code"] == "acceptance_proposal_only"
    assert results[0]["tool_executed"] is False
