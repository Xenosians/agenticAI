"""Learning candidate plans: owner-scoped, path-free, never executable.

These plans back a future authenticated Learning surface. They resolve only
trusted owner-scoped artifact metadata, keep evaluation independent of training
data, and always report the integration gates that are still missing.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys

import pytest

from services.learning_control.contracts import (
    EvidenceCycleRequest,
    LearningArtifact,
    LearningPlanRequest,
    ResourceObservation,
    SourceReview,
    TrainingIdentity,
)
from services.learning_control.planning import prepare_evidence_cycle, prepare_learning_plan

OWNER = "owner-a"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
IDENTITY = TrainingIdentity(model_sha256="1" * 64, tokenizer_sha256="2" * 64,
                            prompt_sha256="3" * 64, catalog_sha256="4" * 64)
REVIEWED = SourceReview(source_id="corpus-1", kind="dataset", license="CC-BY-4.0",
                        reviewed=True, training_eligible=True, review_id="review-1")
NOW = datetime(2026, 10, 28, 4, 0, tzinfo=timezone.utc)
INTEGRATION_GATES = {"trainer_data_admission_required", "host_dispatch_not_integrated",
                     "fixed_evaluation_required", "separate_promotion_required"}


def digest(n: int) -> str:
    return f"{n:064x}"


def artifact(resource_id, kind, *, owner=OWNER, sha=None, records=(), sources=(), identity=IDENTITY, **extra):
    return LearningArtifact(resource_id=resource_id, owner_id=owner, kind=kind,
                            sha256=sha or digest(abs(hash(resource_id)) % (16 ** 12)),
                            identity=identity, sources=tuple(sources),
                            record_sha256s=tuple(records), **extra)


class Registry:
    """Owner-scoped trusted registry double; records every lookup."""

    def __init__(self, *rows):
        self.rows = {(row.owner_id, row.resource_id): row for row in rows}
        self.lookups = []

    def resolve(self, owner_id, resource_id):
        self.lookups.append((owner_id, resource_id))
        return self.rows[(owner_id, resource_id)]


def sft_registry(*, train_records=(digest(1), digest(2)), eval_records=(digest(9),), sources=(REVIEWED,),
                 owner=OWNER, configuration=None):
    return Registry(
        artifact("base-model", "model", owner=owner),
        configuration or artifact("sft-recipe", "adaptation_recipe", owner=owner, records=train_records,
                                  sources=sources, method="sft", role="specialist", tuning="qlora"),
        artifact("eval-suite", "evaluation_suite", owner=owner, records=eval_records),
    )


SFT = LearningPlanRequest(method="sft", role="specialist", tuning="qlora",
                          inputs={"model": "base-model", "configuration": "sft-recipe", "evaluation": "eval-suite"})


def observation(**changes):
    values = dict(method="sft", role="specialist", tuning="qlora", model_sha256=IDENTITY.model_sha256,
                  configuration_sha256=sft_registry().rows[(OWNER, "sft-recipe")].sha256,
                  observed_at=NOW - timedelta(seconds=30), available_free_vram_gib=3.0,
                  required_free_vram_gib=2.5)
    values.update(changes)
    return ResourceObservation(**values)


def test_a_reviewed_sft_plan_is_prepared_but_never_executable():
    plan = prepare_learning_plan(OWNER, SFT, sft_registry(), observation=observation(), as_of=NOW)
    assert plan.status == "prepared"
    assert plan.entrypoint == "scripts/train_recipe.py"
    assert plan.resource_state == "observed_sufficient"
    assert set(plan.inputs) == {"model", "configuration", "evaluation"}
    # Metadata passing is not readiness: the real integration gates stay listed.
    assert INTEGRATION_GATES <= set(plan.blockers)
    assert (plan.execution_enabled, plan.model_loaded, plan.live_tool_execution, plan.promotion_enabled) == (
        False, False, False, False)


def test_without_a_trusted_resource_observation_the_resource_state_is_unknown():
    plan = prepare_learning_plan(OWNER, SFT, sft_registry())
    assert plan.resource_state == "unknown"
    assert "resource_unknown" in plan.blockers


@pytest.mark.parametrize(("changes", "state"), [
    ({"observed_at": NOW - timedelta(minutes=6)}, "stale"),
    ({"observed_at": NOW + timedelta(seconds=5)}, "stale"),
    ({"available_free_vram_gib": 1.5}, "insufficient"),
])
def test_stale_or_insufficient_resources_block_the_plan(changes, state):
    plan = prepare_learning_plan(OWNER, SFT, sft_registry(), observation=observation(**changes), as_of=NOW)
    assert plan.resource_state == state
    assert f"resource_{state}" in plan.blockers


def test_an_observation_for_another_model_or_configuration_is_refused():
    with pytest.raises(ValueError, match="learning_resource_observation_mismatch"):
        prepare_learning_plan(OWNER, SFT, sft_registry(), observation=observation(model_sha256="f" * 64), as_of=NOW)
    with pytest.raises(ValueError, match="learning_observation_time_required"):
        prepare_learning_plan(OWNER, SFT, sft_registry(), observation=observation())


def test_another_owners_resource_is_indistinguishable_from_a_missing_one():
    foreign = sft_registry(owner="owner-b")
    with pytest.raises(PermissionError, match="learning_resource_unavailable") as foreign_error:
        prepare_learning_plan(OWNER, SFT, foreign)
    with pytest.raises(PermissionError, match="learning_resource_unavailable") as missing_error:
        prepare_learning_plan(OWNER, SFT, Registry())
    assert str(foreign_error.value) == str(missing_error.value)
    # Lookups are always made for the session owner, never a body-supplied owner.
    assert {owner for owner, _ in foreign.lookups} == {OWNER}


def test_a_registry_row_claiming_another_owner_or_id_is_refused():
    class Lying(Registry):
        def resolve(self, owner_id, resource_id):
            row = super().resolve(owner_id, resource_id)
            return row.model_copy(update={"owner_id": "owner-b"}) if row.kind == "model" else row

    with pytest.raises(PermissionError, match="learning_resource_unavailable"):
        prepare_learning_plan(OWNER, SFT, Lying(*sft_registry().rows.values()))


def test_an_unvalidated_registry_row_cannot_bypass_the_contract():
    class Smuggling(Registry):
        def resolve(self, owner_id, resource_id):
            row = super().resolve(owner_id, resource_id)
            # model_copy(update=...) skips validation; the planner must revalidate.
            return row.model_copy(update={"sha256": "not-a-digest"}) if row.kind == "model" else row

    with pytest.raises(PermissionError, match="learning_resource_unavailable"):
        prepare_learning_plan(OWNER, SFT, Smuggling(*sft_registry().rows.values()))


def test_owner_is_required():
    with pytest.raises(PermissionError, match="learning_owner_required"):
        prepare_learning_plan(" ", SFT, sft_registry())


@pytest.mark.parametrize("inputs", [
    {"model": "base-model", "configuration": "sft-recipe"},
    {"model": "base-model", "configuration": "sft-recipe", "evaluation": "eval-suite", "extra": "x"},
    {"model": "base-model", "configuration": "base-model", "evaluation": "eval-suite"},
])
def test_inputs_must_match_the_method_contract_exactly(inputs):
    with pytest.raises(ValueError, match="learning_input_contract_mismatch"):
        prepare_learning_plan(OWNER, SFT.model_copy(update={"inputs": inputs}), sft_registry())


def test_a_resource_of_the_wrong_kind_is_refused():
    registry = sft_registry(configuration=artifact("sft-recipe", "dpo_settings", records=(digest(1),),
                                                   sources=(REVIEWED,), method="sft", role="specialist",
                                                   tuning="qlora"))
    with pytest.raises(ValueError, match="learning_resource_kind_mismatch"):
        prepare_learning_plan(OWNER, SFT, registry)


def test_training_and_evaluation_records_must_not_overlap():
    with pytest.raises(ValueError, match="learning_split_membership_overlap"):
        prepare_learning_plan(OWNER, SFT, sft_registry(train_records=(digest(1), digest(9))))


def test_an_empty_evaluation_suite_is_refused():
    with pytest.raises(ValueError, match="learning_independent_evaluation_required"):
        prepare_learning_plan(OWNER, SFT, sft_registry(eval_records=()))


def test_duplicate_or_empty_training_membership_is_refused():
    for records in ((), (digest(1), digest(1))):
        with pytest.raises(ValueError, match="learning_training_membership_invalid"):
            prepare_learning_plan(OWNER, SFT, sft_registry(train_records=records))


def test_artifacts_from_different_base_identities_are_refused():
    other = IDENTITY.model_copy(update={"tokenizer_sha256": "e" * 64})
    registry = sft_registry()
    registry.rows[(OWNER, "eval-suite")] = artifact("eval-suite", "evaluation_suite", records=(digest(9),),
                                                    identity=other)
    with pytest.raises(ValueError, match="learning_training_identity_mismatch"):
        prepare_learning_plan(OWNER, SFT, registry)


def test_a_configuration_for_another_method_is_refused():
    registry = sft_registry(configuration=artifact("sft-recipe", "adaptation_recipe", records=(digest(1),),
                                                   sources=(REVIEWED,), method="sft", role="specialist",
                                                   tuning="lora"))
    with pytest.raises(ValueError, match="learning_configuration_mismatch"):
        prepare_learning_plan(OWNER, SFT, registry)


@pytest.mark.parametrize(("source", "blocker"), [
    (REVIEWED.model_copy(update={"reviewed": False}), "source_review_required"),
    (REVIEWED.model_copy(update={"review_id": None}), "source_review_required"),
    (REVIEWED.model_copy(update={"training_eligible": False}), "source_training_admission_required"),
    (REVIEWED.model_copy(update={"reference_only": True}), "source_training_admission_required"),
    (REVIEWED.model_copy(update={"license": " "}), "source_license_required"),
])
def test_unreviewed_or_reference_only_sources_block_training(source, blocker):
    plan = prepare_learning_plan(OWNER, SFT, sft_registry(sources=(source,)))
    assert plan.status == "blocked"
    assert blocker in plan.blockers


def test_training_data_without_any_reviewed_source_is_blocked():
    plan = prepare_learning_plan(OWNER, SFT, sft_registry(sources=()))
    assert plan.status == "blocked"
    assert "reviewed_source_required" in plan.blockers


def test_method_and_role_follow_the_trusted_method_registry():
    # PPO is Hub-only and QLoRA-only in learning/training/methods.py.
    ppo_for_specialist = LearningPlanRequest(method="ppo", role="specialist", tuning="qlora",
                                             inputs={"model": "base-model"})
    with pytest.raises(ValueError, match="Unsupported training method/role"):
        prepare_learning_plan(OWNER, ppo_for_specialist, sft_registry())


def test_dpo_and_ppo_plans_bind_their_own_entrypoints_and_split_inputs():
    dpo = Registry(
        artifact("base-model", "model"),
        artifact("dpo-settings", "dpo_settings", method="dpo", role="specialist", tuning="qlora"),
        artifact("prefs-train", "preference_train", records=(digest(1), digest(2)), sources=(REVIEWED,)),
        artifact("prefs-valid", "preference_validation", records=(digest(3),), sources=(REVIEWED,)),
        artifact("eval-suite", "evaluation_suite", records=(digest(9),)),
    )
    plan = prepare_learning_plan(OWNER, LearningPlanRequest(
        method="dpo", role="specialist", tuning="qlora",
        inputs={"model": "base-model", "configuration": "dpo-settings", "train": "prefs-train",
                "validation": "prefs-valid", "evaluation": "eval-suite"}), dpo)
    assert plan.entrypoint == "scripts/train_preferences.py"
    assert plan.status == "prepared"

    # Preference train and validation partitions must be disjoint too.
    dpo.rows[(OWNER, "prefs-valid")] = artifact("prefs-valid", "preference_validation", records=(digest(2),),
                                                sources=(REVIEWED,))
    with pytest.raises(ValueError, match="learning_split_membership_overlap"):
        prepare_learning_plan(OWNER, LearningPlanRequest(
            method="dpo", role="specialist", tuning="qlora",
            inputs={"model": "base-model", "configuration": "dpo-settings", "train": "prefs-train",
                    "validation": "prefs-valid", "evaluation": "eval-suite"}), dpo)

    ppo = Registry(
        artifact("base-model", "model"),
        artifact("ppo-settings", "ppo_settings", method="ppo", role="hub", tuning="qlora"),
        artifact("sft-adapter", "adapter_checkpoint"),
        artifact("episodes", "simulator_episodes", records=(digest(5),), sources=(REVIEWED,)),
        artifact("eval-suite", "evaluation_suite", records=(digest(9),)),
    )
    plan = prepare_learning_plan(OWNER, LearningPlanRequest(
        method="ppo", role="hub", tuning="qlora",
        inputs={"model": "base-model", "configuration": "ppo-settings", "checkpoint": "sft-adapter",
                "episodes": "episodes", "evaluation": "eval-suite"}), ppo)
    assert plan.entrypoint == "scripts/train_ppo.py"
    assert plan.live_tool_execution is False


def test_the_plan_hash_changes_when_any_bound_input_changes():
    first = prepare_learning_plan(OWNER, SFT, sft_registry())
    again = prepare_learning_plan(OWNER, SFT, sft_registry())
    changed = prepare_learning_plan(OWNER, SFT, sft_registry(train_records=(digest(1), digest(3))))
    assert first.plan_sha256 == again.plan_sha256
    assert first.plan_sha256 != changed.plan_sha256
    assert first.inputs["configuration"].membership_sha256 != changed.inputs["configuration"].membership_sha256


def test_requests_cannot_carry_paths_or_extra_fields():
    with pytest.raises(ValueError):
        LearningPlanRequest(method="sft", role="specialist", tuning="qlora",
                            inputs={"model": "../../models/base"})
    with pytest.raises(ValueError):
        LearningPlanRequest(method="sft", role="specialist", tuning="qlora",
                            inputs={"model": "base-model"}, owner_id="owner-b")


def test_evidence_cycle_binds_the_four_reviewed_ledgers_without_collecting_or_training():
    registry = Registry(
        artifact("trajectories", "trajectory_ledger", records=(digest(1),)),
        artifact("corrections", "correction_ledger", records=(digest(2),)),
        artifact("reviews", "review_ledger", records=(digest(3),)),
        artifact("context", "context_ledger", records=(digest(4),)),
    )
    plan = prepare_evidence_cycle(OWNER, EvidenceCycleRequest(inputs={
        "trajectories": "trajectories", "corrections": "corrections", "reviews": "reviews",
        "context": "context"}), registry)
    assert plan.entrypoint == "learning/cli/run_continual_cycle.py"
    assert set(plan.inputs) == {"trajectories", "corrections", "reviews", "context"}
    assert (plan.execution_enabled, plan.training_enabled, plan.automatic_promotion,
            plan.web_collection_enabled) == (False, False, False, False)


def test_importing_the_package_loads_no_trainer_model_or_configuration():
    # A fresh interpreter: in the full suite torch/config are already imported.
    probe = ("import sys, services.learning_control, services.learning_control.contracts, "
             "services.learning_control.planning; "
             "print(sorted({m.split('.')[0] for m in sys.modules} & "
             "{'torch', 'transformers', 'peft', 'bitsandbytes', 'datasets', 'config', 'httpx'}))")
    result = subprocess.run([sys.executable, "-c", probe], cwd=PROJECT_ROOT, capture_output=True, text=True,
                            timeout=60, check=True)
    assert result.stdout.strip() == "[]"
