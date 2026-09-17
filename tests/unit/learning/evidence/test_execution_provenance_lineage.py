import json

from pathlib import (
    Path,
)

from learning.curation.preferences import (
    build_preference_example,
)

from learning.datasets.dataset_loader import (
    PreferenceDatasetLoader,
)

from learning.datasets.records import (
    PreferenceDatasetBuilder,
)

from learning.evidence.execution_provenance import (
    SpecialistExecutionProvenance,
)

from learning.evidence.types import (
    CorrectionEvent,
    CorrectionValue,
    ExecutionReward,
    LearningTrajectory,
    TrajectorySignals,
    TrajectoryStep,
)


HASH_A = (
    "a"
    * 64
)


def execution_provenance(
) -> SpecialistExecutionProvenance:

    return (
        SpecialistExecutionProvenance(
            provenance_complete=True,

            agent_name=(
                "account-specialist"
            ),

            model_key=(
                "qwen2.5-0.5b-funccall"
            ),

            backend=(
                "qwen-funccall"
            ),

            quantization=(
                "bnb4"
            ),

            compute_dtype=(
                "bfloat16"
            ),

            device_map=(
                "auto"
            ),

            bnb_4bit_quant_type=(
                "nf4"
            ),

            bnb_4bit_use_double_quant=True,

            model_profile_sha256=(
                HASH_A
            ),

            model_artifact_sha256=(
                HASH_A
            ),

            model_weights_sha256=(
                HASH_A
            ),

            model_config_sha256=(
                HASH_A
            ),

            generation_config_sha256=(
                HASH_A
            ),

            tokenizer_artifact_sha256=(
                HASH_A
            ),

            tokenizer_config_sha256=(
                HASH_A
            ),

            tokenizer_json_sha256=(
                HASH_A
            ),

            chat_template_sha256=(
                HASH_A
            ),

            agent_definition_sha256=(
                HASH_A
            ),

            capability_catalog_sha256=(
                HASH_A
            ),

            system_prompt_sha256=(
                HASH_A
            ),

            messages_sha256=(
                HASH_A
            ),

            user_request_sha256=(
                HASH_A
            ),

            task_instructions_sha256=(
                HASH_A
            ),

            max_new_tokens=256,
        )
    )


def trajectory(
    *,
    provenance: (
        SpecialistExecutionProvenance
        | None
    ),
) -> LearningTrajectory:

    return (
        LearningTrajectory(
            trajectory_id=(
                "trajectory-1"
            ),

            observed_at=(
                "2026-09-17T00:00:00+00:00"
            ),

            job_id=(
                "job-1"
            ),

            attempt=1,

            user_request=(
                "Is jdoe locked?"
            ),

            hub_model=(
                "hub-main"
            ),

            hub_status=(
                "success"
            ),

            routes=[
                "account-specialist"
            ],

            steps=[
                TrajectoryStep(
                    task_id=(
                        "task-1"
                    ),

                    task_instructions=(
                        "Check the requested account."
                    ),

                    execution_provenance=(
                        provenance
                    ),

                    agent=(
                        "account-specialist"
                    ),

                    status=(
                        "success"
                    ),

                    outcome_code=(
                        "success"
                    ),

                    proposed_tool=(
                        "account_status"
                    ),

                    proposed_arguments={
                        "user_id":
                            "wrong-user",
                    },

                    tool_result={
                        "ok":
                            True,
                    },
                )
            ],

            final_answer=(
                "wrong-user is unlocked."
            ),

            signals=(
                TrajectorySignals(
                    delegated=True,

                    route_count=1,

                    specialist_count=1,

                    specialist_success_count=1,

                    tool_proposed_count=1,

                    tool_success_count=1,

                    overall_success=True,
                )
            ),

            execution_reward=(
                ExecutionReward(
                    total=1.0,
                    quality_eligible=True,
                )
            ),

            dataset_eligible=False,
        )
    )


def correction(
) -> CorrectionEvent:

    return (
        CorrectionEvent(
            correction_id=(
                "correction-1"
            ),

            observed_at=(
                "2026-09-17T00:01:00+00:00"
            ),

            trajectory_id=(
                "trajectory-1"
            ),

            task_id=(
                "task-1"
            ),

            correction_type=(
                "arguments"
            ),

            source=(
                "explicit_user"
            ),

            values=[
                CorrectionValue(
                    field=(
                        "user_id"
                    ),

                    rejected_value=(
                        "wrong-user"
                    ),

                    chosen_value=(
                        "jdoe"
                    ),
                )
            ],

            dataset_eligible=True,
        )
    )


def test_execution_provenance_survives_preference_derivation():

    original = (
        execution_provenance()
    )

    source_trajectory = (
        trajectory(
            provenance=(
                original
            )
        )
    )

    example = (
        build_preference_example(
            trajectory=(
                source_trajectory
            ),

            correction=(
                correction()
            ),
        )
    )

    assert (
        example.execution_provenance
        == original
    )

    assert (
        example.execution_provenance
        is not original
    )

    assert (
        example.execution_provenance
        is not (
            source_trajectory
            .steps[
                0
            ]
            .execution_provenance
        )
    )

    assert (
        example
        .execution_provenance
        .model_artifact_sha256
        == HASH_A
    )

    assert (
        example
        .task_instructions
        == "Check the requested account."
    )


def test_execution_provenance_survives_dataset_promotion_and_reload(
    tmp_path: Path,
):

    original = (
        execution_provenance()
    )

    example = (
        build_preference_example(
            trajectory=(
                trajectory(
                    provenance=(
                        original
                    )
                )
            ),

            correction=(
                correction()
            ),
        )
    )

    dataset_root = (
        tmp_path
        / "datasets"
    )

    builder = (
        PreferenceDatasetBuilder(
            root=(
                dataset_root
            ),

            eval_paths=[],
        )
    )

    manifest = (
        builder.promote(
            examples=[
                example
            ],

            promoted_by=(
                "trusted_review"
            ),

            promotion_reason=(
                "Verified provenance lineage."
            ),
        )
    )

    loader = (
        PreferenceDatasetLoader(
            root=(
                dataset_root
            )
        )
    )

    (
        loaded_manifest,
        records,
    ) = (
        loader.load(
            manifest.version
        )
    )

    assert (
        loaded_manifest.version
        == manifest.version
    )

    assert (
        len(
            records
        )
        == 1
    )

    loaded = (
        records[
            0
        ]
    )

    assert (
        loaded.execution_provenance
        == original
    )

    assert (
        loaded
        .execution_provenance
        .model_artifact_sha256
        == HASH_A
    )

    raw_record_path = (
        dataset_root
        / "preference"
        / manifest.version
        / "records.jsonl"
    )

    raw = (
        json.loads(
            raw_record_path
            .read_text(
                encoding="utf-8"
            )
            .splitlines()[
                0
            ]
        )
    )

    assert (
        "execution_provenance"
        in raw
    )

    assert (
        raw[
            "execution_provenance"
        ][
            "model_artifact_sha256"
        ]
        == HASH_A
    )


def test_legacy_evidence_without_provenance_remains_readable(
    tmp_path: Path,
):

    example = (
        build_preference_example(
            trajectory=(
                trajectory(
                    provenance=None
                )
            ),

            correction=(
                correction()
            ),
        )
    )

    assert (
        example.execution_provenance
        is None
    )

    dataset_root = (
        tmp_path
        / "legacy-datasets"
    )

    builder = (
        PreferenceDatasetBuilder(
            root=(
                dataset_root
            ),

            eval_paths=[],
        )
    )

    manifest = (
        builder.promote(
            examples=[
                example
            ],

            promoted_by=(
                "trusted_review"
            ),

            promotion_reason=(
                "Legacy compatibility test."
            ),
        )
    )

    loader = (
        PreferenceDatasetLoader(
            root=(
                dataset_root
            )
        )
    )

    (
        _manifest,
        records,
    ) = (
        loader.load(
            manifest.version
        )
    )

    assert (
        len(
            records
        )
        == 1
    )

    assert (
        records[
            0
        ]
        .execution_provenance
        is None
    )