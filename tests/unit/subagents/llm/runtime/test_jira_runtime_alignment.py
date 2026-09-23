from pathlib import Path

from subagents.core.definitions.loader import (
    load_agent_definition,
)


def test_jira_eval_uses_runtime_scheduler_stack_and_configured_profile():
    source = (
        Path(
            "scripts/jira_specialist_model_eval.py"
        )
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "ModelManager"
        in source
    )

    assert (
        "GpuScheduler"
        in source
    )

    assert (
        "InferenceCoordinator"
        in source
    )

    assert (
        "InferencePriority.SPECIALIST"
        in source
    )

    assert (
        ".require_model_profile("
        in source
    )

    assert (
        "ModelProfileSettings("
        not in source
    )

    assert (
        "HFCausalWorkerBackend("
        not in source
    )

    assert (
        "/Models/BLOOMZ-560M"
        not in source
    )


def test_jira_eval_emits_learning_provenance_and_candidate_report():
    source = (
        Path(
            "scripts/jira_specialist_model_eval.py"
        )
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "build_specialist_execution_provenance"
        in source
    )

    assert (
        "SpecialistModelCaseResult"
        in source
    )

    assert (
        "write_specialist_model_report"
        in source
    )

    assert (
        ".runtime"
        in source
    )


def test_jira_production_model_is_not_auto_promoted_by_candidate_alignment():
    agent = (
        load_agent_definition(
            "subagents/agents/jira-specialist.md"
        )
    )

    assert (
        agent.model
        == "hub-main"
    )
