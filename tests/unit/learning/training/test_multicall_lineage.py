from __future__ import annotations

import asyncio
import json

from dataclasses import (
    replace,
)

from pathlib import (
    Path,
)

from learning.curation.corrections import (
    CorrectionRecorder,
)

from learning.curation.preferences import (
    build_preference_example,
)

from learning.datasets.records import (
    PreferenceDatasetBuilder,
)

from learning.evidence.recorder import (
    TrajectoryRecorder,
)

from learning.evidence.types import (
    CorrectionEvent,
    LearningTrajectory,
    PreferenceDatasetRecord,
)

from subagents.core.definitions.registry import (
    AgentRegistry,
)

from subagents.core.definitions.types import (
    AgentTask,
    HubResult,
)

from subagents.core.orchestration.runtime import (
    AgentRuntime,
)

from tests.unit.learning.training.test_dpo_materializer import (
    materializer,
)


USER_REQUEST = (
    "Check jdoe's account status, "
    "not alice's."
)


TASK_INSTRUCTIONS = (
    "Check the account status for jdoe only."
)


REJECTED_CALLS = [
    {
        "name":
            "account_status",

        "arguments": {
            "user_id":
                "jdoe",
        },
    },

    {
        "name":
            "unlock_user",

        "arguments": {
            "user_id":
                "alice",
        },
    },
]


CHOSEN_CALLS = [
    {
        "name":
            "account_status",

        "arguments": {
            "user_id":
                "jdoe",
        },
    },
]


RAW_MODEL_OUTPUT = (
    json.dumps(
        REJECTED_CALLS,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    )
)


class FakeInference:
    def __init__(
        self,
        response: str,
    ) -> None:

        self.response = (
            response
        )

    async def generate(
        self,
        *,
        model_key,
        messages,
        max_new_tokens,
        priority,
    ):

        return (
            self.response
        )


class NeverExecuteGateway:
    """
    Multi-call runtime output must be rejected before ToolGateway.

    If this fake is called, the safety contract regressed.
    """

    async def execute(
        self,
        agent,
        user_input,
        tool_name,
        arguments,
    ):

        raise AssertionError(
            "ToolGateway must not execute "
            "an invalid multi-call worker response."
        )


def read_one_jsonl(
    path: Path,
) -> dict:

    lines = (
        path
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    assert (
        len(
            lines
        )
        == 1
    )

    return (
        json.loads(
            lines[
                0
            ]
        )
    )


def test_multicall_behavior_survives_complete_learning_lineage(
    tmp_path: Path,
):

    # ========================================================
    # TARGET TRAINING ENVIRONMENT
    #
    # Reuse the existing synthetic model + agent fixture so the
    # runtime capture and DPO provenance gate observe the exact
    # same model/agent/tool environment.
    # ========================================================

    (
        dpo_materializer,
        profile,
        _,
    ) = (
        materializer(
            tmp_path
        )
    )

    # ========================================================
    # REAL RUNTIME SHAPE
    # ========================================================

    registry = (
        AgentRegistry()
    )

    registry.register(
        dpo_materializer.agent
    )

    runtime = (
        AgentRuntime(
            agent_registry=(
                registry
            ),

            inference=(
                FakeInference(
                    RAW_MODEL_OUTPUT
                )
            ),

            tool_gateway=(
                NeverExecuteGateway()
            ),

            model_profile_resolver=(
                lambda model_key:
                    profile
            ),
        )
    )

    task = (
        AgentTask(
            task_id=(
                "task-multicall-lineage"
            ),

            agent_name=(
                "account-specialist"
            ),

            user_request=(
                USER_REQUEST
            ),

            instructions=(
                TASK_INSTRUCTIONS
            ),
        )
    )

    runtime_result = (
        asyncio.run(
            runtime.run(
                task
            )
        )
    )

    assert (
        runtime_result.status
        == "error"
    )

    assert (
        runtime_result.outcome_code
        == "invalid_tool_call_count"
    )

    assert (
        runtime_result.raw_model_output
        == RAW_MODEL_OUTPUT
    )

    assert (
        runtime_result.proposed_tool_calls
        == REJECTED_CALLS
    )

    assert (
        runtime_result.proposed_tool
        is None
    )

    assert (
        runtime_result.proposed_arguments
        is None
    )

    assert (
        task.execution_provenance
        is not None
    )

    assert (
        task.execution_provenance[
            "provenance_complete"
        ]
        is True
    )

    # ========================================================
    # ORCHESTRATOR ENRICHMENT SHAPE
    #
    # The real orchestrator attaches task context and provenance
    # after specialist execution via dataclasses.replace.
    #
    # Reproduce that exact boundary here.
    # ========================================================

    enriched_result = (
        replace(
            runtime_result,

            task_instructions=(
                TASK_INSTRUCTIONS
            ),

            execution_provenance=(
                task.execution_provenance
            ),
        )
    )

    hub_result = (
        HubResult(
            status=(
                "partial_error"
            ),

            user_request=(
                USER_REQUEST
            ),

            routes=[
                "account-specialist",
            ],

            results=[
                enriched_result
            ],

            answer=(
                "The specialist returned "
                "an invalid tool-call count."
            ),
        )
    )

    # ========================================================
    # DURABLE TRAJECTORY
    # ========================================================

    trajectory_path = (
        tmp_path
        / "runtime"
        / "trajectories.jsonl"
    )

    trajectory_recorder = (
        TrajectoryRecorder(
            path=(
                trajectory_path
            ),

            enabled=True,

            hub_model=(
                "hub-main"
            ),
        )
    )

    trajectory_payload = (
        trajectory_recorder.record(
            job_id=(
                "job-multicall-lineage"
            ),

            attempt=1,

            result=(
                hub_result
            ),
        )
    )

    assert (
        trajectory_payload
        is not None
    )

    trajectory = (
        LearningTrajectory
        .model_validate(
            trajectory_payload
        )
    )

    assert (
        len(
            trajectory.steps
        )
        == 1
    )

    step = (
        trajectory.steps[
            0
        ]
    )

    assert (
        step.raw_model_output
        == RAW_MODEL_OUTPUT
    )

    assert (
        step.proposed_tool_calls
        == REJECTED_CALLS
    )

    assert (
        step.execution_provenance
        is not None
    )

    assert (
        step
        .execution_provenance
        .provenance_complete
        is True
    )

    # ========================================================
    # TRUSTED CORRECTION
    #
    # Correct the complete observed call set.
    #
    # Never synthesize a singular rejected tool here.
    # ========================================================

    correction_path = (
        tmp_path
        / "runtime"
        / "corrections.jsonl"
    )

    correction_recorder = (
        CorrectionRecorder(
            path=(
                correction_path
            ),

            enabled=True,
        )
    )

    correction_payload = (
        correction_recorder.record(
            trajectory_id=(
                trajectory.trajectory_id
            ),

            task_id=(
                step.task_id
            ),

            correction_type=(
                "tool_selection"
            ),

            source=(
                "trusted_review"
            ),

            note=(
                "The worker emitted multiple calls "
                "despite the request targeting jdoe only."
            ),

            values=[
                {
                    "field":
                        "tool_calls",

                    "rejected_value":
                        REJECTED_CALLS,

                    "chosen_value":
                        CHOSEN_CALLS,
                }
            ],
        )
    )

    assert (
        correction_payload
        is not None
    )

    correction = (
        CorrectionEvent
        .model_validate(
            correction_payload
        )
    )

    # ========================================================
    # PREFERENCE EXAMPLE
    # ========================================================

    preference = (
        build_preference_example(
            trajectory=(
                trajectory
            ),

            correction=(
                correction
            ),
        )
    )

    assert (
        preference.rejected.tool_calls
        == REJECTED_CALLS
    )

    assert (
        preference.chosen.tool_calls
        == CHOSEN_CALLS
    )

    assert (
        preference.rejected.tool
        is None
    )

    assert (
        preference.chosen.tool
        is None
    )

    assert (
        preference.execution_provenance
        is not None
    )

    assert (
        preference
        .execution_provenance
        .provenance_complete
        is True
    )

    # ========================================================
    # IMMUTABLE PREFERENCE DATASET
    # ========================================================

    dataset_root = (
        tmp_path
        / "datasets"
    )

    dataset_builder = (
        PreferenceDatasetBuilder(
            root=(
                dataset_root
            ),

            # Synthetic lineage test.
            #
            # Held-out contamination behavior is tested elsewhere.
            eval_paths=[],
        )
    )

    manifest = (
        dataset_builder.promote(
            examples=[
                preference
            ],

            promoted_by=(
                "trusted_review"
            ),

            promotion_reason=(
                "Verified synthetic multi-call "
                "lineage integration test."
            ),
        )
    )

    assert (
        manifest.record_count
        == 1
    )

    dataset_record_payload = (
        read_one_jsonl(
            dataset_root
            / "preference"
            / manifest.version
            / "records.jsonl"
        )
    )

    dataset_record = (
        PreferenceDatasetRecord
        .model_validate(
            dataset_record_payload
        )
    )

    assert (
        dataset_record.rejected.tool_calls
        == REJECTED_CALLS
    )

    assert (
        dataset_record.chosen.tool_calls
        == CHOSEN_CALLS
    )

    assert (
        dataset_record.execution_provenance
        is not None
    )

    assert (
        dataset_record
        .execution_provenance
        .provenance_complete
        is True
    )

    # ========================================================
    # DPO MATERIALIZATION
    #
    # The exact captured rejected behavior must survive as a
    # canonical JSON completion.
    #
    # The corrected chosen behavior must contain exactly one
    # allowed call.
    # ========================================================

    dpo_result = (
        dpo_materializer.build(
            records=[
                dataset_record
            ],

            source_split_id=(
                "multicall-lineage"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                manifest.content_sha256
            ),
        )
    )

    assert (
        dpo_result.manifest.record_count
        == 1
    )

    dpo_payload = (
        read_one_jsonl(
            dpo_result.output_directory
            / "records.jsonl"
        )
    )

    assert (
        dpo_payload[
            "change_type"
        ]
        == "tool_calls"
    )

    rejected_completion = (
        json.loads(
            dpo_payload[
                "rejected"
            ]
        )
    )

    chosen_completion = (
        json.loads(
            dpo_payload[
                "chosen"
            ]
        )
    )

    assert (
        rejected_completion
        == REJECTED_CALLS
    )

    assert (
        chosen_completion
        == CHOSEN_CALLS
    )

    assert (
        len(
            rejected_completion
        )
        == 2
    )

    assert (
        len(
            chosen_completion
        )
        == 1
    )

    assert (
        dpo_payload[
            "source_trajectory_id"
        ]
        == trajectory.trajectory_id
    )

    assert (
        dpo_payload[
            "source_correction_id"
        ]
        == correction.correction_id
    )

    assert (
        dpo_payload[
            "source_execution_provenance"
        ][
            "provenance_complete"
        ]
        is True
    )

    # ========================================================
    # FINAL LINEAGE INVARIANTS
    # ========================================================

    assert (
        trajectory
        .steps[
            0
        ]
        .proposed_tool_calls
        ==
        preference
        .rejected
        .tool_calls
        ==
        dataset_record
        .rejected
        .tool_calls
        ==
        rejected_completion
    )

    assert (
        preference
        .chosen
        .tool_calls
        ==
        dataset_record
        .chosen
        .tool_calls
        ==
        chosen_completion
    )
