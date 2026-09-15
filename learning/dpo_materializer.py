from __future__ import annotations

import hashlib
import json
import shutil

from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from learning.types import (
    PreferenceDatasetRecord,
)

from subagents.core.loader import (
    load_agent_definition,
)

from subagents.core.tool_prompt import (
    build_worker_system_prompt,
)


# ============================================================
# CONSTANTS
# ============================================================


SUPPORTED_SPECIALIST_CHANGE_TYPES = {
    "tool",
    "arguments",
}


# ============================================================
# HELPERS
# ============================================================


def _utc_now(
) -> str:

    return (
        datetime
        .now(
            timezone.utc
        )
        .isoformat()
    )


def _canonical_json(
    value,
) -> str:

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
    )


def _sha256_text(
    value: str,
) -> str:

    return (
        hashlib
        .sha256(
            value.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


def _tool_call_response(
    *,
    tool: str,
    arguments: dict,
) -> str:
    """
    Serialize one specialist response using the exact runtime
    worker contract:

        [
          {
            "name": "...",
            "arguments": {...}
          }
        ]

    Stable canonical JSON is used for training artifacts.
    """

    return (
        _canonical_json(
            [
                {
                    "name":
                        tool,

                    "arguments":
                        arguments,
                }
            ]
        )
    )


# ============================================================
# MATERIALIZED RECORD
# ============================================================


class SpecialistDpoRecord(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "specialist-dpo-record.v1"
        ),
        alias="schema",
    )

    dpo_record_id: str

    source_record_id: str

    source_trajectory_id: str

    source_correction_id: str

    target_agent: str

    target_model_key: str

    change_type: str

    prompt_messages: list[
        dict[
            str,
            str,
        ]
    ]

    chosen: str

    rejected: str


class SpecialistDpoManifest(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "specialist-dpo-manifest.v1"
        ),
        alias="schema",
    )

    created_at: str

    target_agent: str

    target_model_key: str

    source_split_id: str

    source_partition: str

    source_sha256: str

    record_count: int

    excluded_record_count: int

    exclusion_reason_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    content_sha256: str

    source_record_ids: list[
        str
    ] = Field(
        default_factory=list
    )


class SpecialistDpoBuildResult(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "specialist-dpo-build-result.v1"
        ),
        alias="schema",
    )

    manifest: SpecialistDpoManifest

    output_directory: Path


# ============================================================
# MATERIALIZER
# ============================================================


class SpecialistDpoMaterializer:
    """
    Convert verified Phase-3 preference records into a
    target-specific specialist DPO artifact.

    Only specialist behavior is accepted here.

    Supported learning dimensions:

        tool selection
        tool arguments

    Deliberately excluded:

        route / agent corrections
            -> belong to Hub/router training

        final-answer corrections
            -> belong to Hub response training

    Mixed corrections are rejected from this target dataset rather
    than being silently interpreted.
    """

    def __init__(
        self,
        *,
        agent_definition_path: Path,
        output_root: Path,
    ) -> None:

        self.agent_definition_path = (
            agent_definition_path
            .expanduser()
            .resolve()
        )

        self.output_root = (
            output_root
            .expanduser()
            .resolve()
        )

        self.agent = (
            load_agent_definition(
                self.agent_definition_path
            )
        )

    # ========================================================
    # PROMPT
    # ========================================================

    def _build_prompt_messages(
        self,
        record: PreferenceDatasetRecord,
    ) -> list[
        dict[
            str,
            str,
        ]
    ]:

        messages = [
            {
                "role":
                    "system",

                "content":
                    build_worker_system_prompt(
                        self.agent
                    ),
            },

            {
                "role":
                    "user",

                "content":
                    record.user_request,
            },
        ]

        if (
            record.task_instructions
            is not None
        ):

            normalized = (
                record
                .task_instructions
                .strip()
            )

            if normalized:

                messages.append(
                    {
                        "role":
                            "user",

                        "content":
                            (
                                "Additional task context "
                                "from the routing stage:\n"
                                f"{normalized}"
                            ),
                    }
                )

        return messages

    # ========================================================
    # TARGET CLASSIFICATION
    # ========================================================

    def _change_dimensions(
        self,
        record: PreferenceDatasetRecord,
    ) -> set[
        str
    ]:

        dimensions: set[
            str
        ] = set()

        if (
            record.rejected.agent
            != record.chosen.agent
        ):

            dimensions.add(
                "agent"
            )

        if (
            record.rejected.tool
            != record.chosen.tool
        ):

            dimensions.add(
                "tool"
            )

        if (
            record.rejected.arguments
            != record.chosen.arguments
        ):

            dimensions.add(
                "arguments"
            )

        if (
            record.rejected.answer
            != record.chosen.answer
        ):

            dimensions.add(
                "answer"
            )

        return dimensions

    def _classify(
        self,
        record: PreferenceDatasetRecord,
    ) -> tuple[
        str | None,
        str | None,
    ]:
        """
        Return:

            (change_type, exclusion_reason)

        Exactly one specialist learning dimension is permitted.
        """

        dimensions = (
            self._change_dimensions(
                record
            )
        )

        if not dimensions:

            return (
                None,
                "no_behavior_change",
            )

        if (
            "agent"
            in dimensions
        ):

            return (
                None,
                "router_target",
            )

        if (
            "answer"
            in dimensions
        ):

            return (
                None,
                "hub_answer_target",
            )

        specialist_dimensions = (
            dimensions
            & SUPPORTED_SPECIALIST_CHANGE_TYPES
        )

        if (
            len(
                specialist_dimensions
            )
            != 1
        ):

            return (
                None,
                "mixed_specialist_change",
            )

        change_type = (
            next(
                iter(
                    specialist_dimensions
                )
            )
        )

        return (
            change_type,
            None,
        )

    # ========================================================
    # TARGET VALIDATION
    # ========================================================

    def _validate_specialist_target(
        self,
        record: PreferenceDatasetRecord,
    ) -> str | None:
        """
        Return an exclusion reason when this record does not belong
        to the selected specialist.
        """

        rejected_agent = (
            record.rejected.agent
        )

        chosen_agent = (
            record.chosen.agent
        )

        if (
            rejected_agent
            != self.agent.name
            or chosen_agent
            != self.agent.name
        ):

            return (
                "different_specialist"
            )

        if (
            record.rejected.tool
            is None
            or record.chosen.tool
            is None
        ):

            return (
                "missing_tool"
            )

        if (
            record.rejected.arguments
            is None
            or record.chosen.arguments
            is None
        ):

            return (
                "missing_arguments"
            )

        if (
            record.rejected.tool
            not in self.agent.tools
        ):

            return (
                "rejected_tool_not_allowed"
            )

        if (
            record.chosen.tool
            not in self.agent.tools
        ):

            return (
                "chosen_tool_not_allowed"
            )

        return None

    # ========================================================
    # RECORD BUILDING
    # ========================================================

    def _build_record(
        self,
        *,
        source: PreferenceDatasetRecord,
        change_type: str,
    ) -> SpecialistDpoRecord:

        prompt_messages = (
            self._build_prompt_messages(
                source
            )
        )

        chosen = (
            _tool_call_response(
                tool=(
                    source.chosen.tool
                ),

                arguments=(
                    source.chosen.arguments
                    or {}
                ),
            )
        )

        rejected = (
            _tool_call_response(
                tool=(
                    source.rejected.tool
                ),

                arguments=(
                    source.rejected.arguments
                    or {}
                ),
            )
        )

        identity_payload = {
            "source_record_id":
                source.record_id,

            "target_agent":
                self.agent.name,

            "target_model_key":
                self.agent.model,

            "change_type":
                change_type,

            "prompt_messages":
                prompt_messages,

            "chosen":
                chosen,

            "rejected":
                rejected,
        }

        dpo_record_id = (
            "dpo-"
            + _sha256_text(
                _canonical_json(
                    identity_payload
                )
            )[
                :24
            ]
        )

        return (
            SpecialistDpoRecord(
                dpo_record_id=(
                    dpo_record_id
                ),

                source_record_id=(
                    source.record_id
                ),

                source_trajectory_id=(
                    source.trajectory_id
                ),

                source_correction_id=(
                    source.correction_id
                ),

                target_agent=(
                    self.agent.name
                ),

                target_model_key=(
                    self.agent.model
                ),

                change_type=(
                    change_type
                ),

                prompt_messages=(
                    prompt_messages
                ),

                chosen=(
                    chosen
                ),

                rejected=(
                    rejected
                ),
            )
        )

    # ========================================================
    # BUILD
    # ========================================================

    def build(
        self,
        *,
        records: list[
            PreferenceDatasetRecord
        ],
        source_split_id: str,
        source_partition: str,
        source_sha256: str,
    ) -> SpecialistDpoBuildResult:

        normalized_partition = (
            source_partition
            .strip()
            .lower()
        )

        if normalized_partition not in {
            "train",
            "validation",
        }:

            raise ValueError(
                "source_partition must be "
                "'train' or 'validation'."
            )

        materialized: list[
            SpecialistDpoRecord
        ] = []

        exclusion_counts: dict[
            str,
            int,
        ] = {}

        for record in records:

            target_error = (
                self._validate_specialist_target(
                    record
                )
            )

            if target_error is not None:

                exclusion_counts[
                    target_error
                ] = (
                    exclusion_counts.get(
                        target_error,
                        0,
                    )
                    + 1
                )

                continue

            (
                change_type,
                exclusion_reason,
            ) = (
                self._classify(
                    record
                )
            )

            if exclusion_reason is not None:

                exclusion_counts[
                    exclusion_reason
                ] = (
                    exclusion_counts.get(
                        exclusion_reason,
                        0,
                    )
                    + 1
                )

                continue

            if change_type is None:

                raise ValueError(
                    "Internal DPO classification error."
                )

            materialized.append(
                self._build_record(
                    source=(
                        record
                    ),

                    change_type=(
                        change_type
                    ),
                )
            )

        if not materialized:

            raise ValueError(
                "No target-compatible specialist DPO "
                "records are available."
            )

        seen_ids: set[
            str
        ] = set()

        for record in materialized:

            if (
                record.dpo_record_id
                in seen_ids
            ):

                raise ValueError(
                    "Duplicate materialized DPO record: "
                    f"{record.dpo_record_id}"
                )

            seen_ids.add(
                record.dpo_record_id
            )

        serialized_records = [
            _canonical_json(
                record.model_dump(
                    mode="json",
                    by_alias=True,
                )
            )

            for record
            in materialized
        ]

        content_blob = (
            "\n".join(
                serialized_records
            )
            + "\n"
        )

        content_sha256 = (
            _sha256_text(
                content_blob
            )
        )

        target_directory = (
            self.output_root
            / source_split_id
            / self.agent.name
            / normalized_partition
        )

        if target_directory.exists():

            raise ValueError(
                "Target-specific DPO artifact already exists: "
                f"{target_directory}"
            )

        target_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        manifest = (
            SpecialistDpoManifest(
                created_at=(
                    _utc_now()
                ),

                target_agent=(
                    self.agent.name
                ),

                target_model_key=(
                    self.agent.model
                ),

                source_split_id=(
                    source_split_id
                ),

                source_partition=(
                    normalized_partition
                ),

                source_sha256=(
                    source_sha256
                ),

                record_count=(
                    len(
                        materialized
                    )
                ),

                excluded_record_count=(
                    len(
                        records
                    )
                    - len(
                        materialized
                    )
                ),

                exclusion_reason_counts=(
                    dict(
                        sorted(
                            exclusion_counts.items()
                        )
                    )
                ),

                content_sha256=(
                    content_sha256
                ),

                source_record_ids=[
                    record.source_record_id

                    for record
                    in materialized
                ],
            )
        )

        try:

            (
                target_directory
                / "records.jsonl"
            ).write_text(
                content_blob,
                encoding="utf-8",
            )

            (
                target_directory
                / "manifest.json"
            ).write_text(
                json.dumps(
                    manifest.model_dump(
                        mode="json",
                        by_alias=True,
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        except Exception:

            shutil.rmtree(
                target_directory,
                ignore_errors=True,
            )

            raise

        return (
            SpecialistDpoBuildResult(
                manifest=(
                    manifest
                ),

                output_directory=(
                    target_directory
                ),
            )
        )