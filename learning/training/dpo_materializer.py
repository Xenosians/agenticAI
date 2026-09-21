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

from config import (
    ModelProfileSettings,
)

from learning.evidence.execution_provenance import (
    SpecialistExecutionProvenance,
)

from learning.evidence.types import (
    PreferenceDatasetRecord,
)

from learning.training.provenance_gate import (
    SPECIALIST_TRAINING_MAX_NEW_TOKENS,
    SpecialistTrainingEnvironment,
    build_specialist_training_environment,
    validate_specialist_training_provenance,
)

from subagents.core.definitions.loader import (
    load_agent_definition,
)


# ============================================================
# CONSTANTS
# ============================================================


SUPPORTED_SPECIALIST_CHANGE_TYPES = {
    "tool",
    "arguments",
    "tool_calls",
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


def _tool_calls_response(
    tool_calls: list[
        dict
    ],
) -> str:
    """
    Serialize an exact specialist worker call set.

    This representation is used for reviewed invalid-cardinality
    negative evidence.

    The call set is preserved exactly rather than normalized into
    one canonical call.
    """

    return (
        _canonical_json(
            tool_calls
        )
    )


def _tool_call_response(
    *,
    tool: str,
    arguments: dict,
) -> str:
    """
    Serialize exactly one specialist tool call using the runtime
    worker contract.
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


def _increment_exclusion(
    counts: dict[
        str,
        int,
    ],
    reason: str,
) -> None:

    counts[
        reason
    ] = (
        counts.get(
            reason,
            0,
        )
        + 1
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

    # Additive compatibility field.
    #
    # New Phase-4C.2 materializations always populate it.
    # Historical synthetic DPO artifacts remain readable until the
    # downstream training gate is hardened in Phase-4C.2-C.
    source_execution_provenance: (
        SpecialistExecutionProvenance
        | None
    ) = None


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

    # ========================================================
    # PHASE-4C.2 PROVENANCE ENFORCEMENT
    #
    # Defaults preserve readability of historical synthetic
    # materializations.
    #
    # New materializations set these fields explicitly.
    # ========================================================

    execution_provenance_enforced: bool = False

    target_model_artifact_sha256: (
        str
        | None
    ) = None

    target_model_weights_sha256: (
        str
        | None
    ) = None

    target_tokenizer_artifact_sha256: (
        str
        | None
    ) = None

    target_model_profile_sha256: (
        str
        | None
    ) = None

    target_agent_definition_sha256: (
        str
        | None
    ) = None

    target_capability_catalog_sha256: (
        str
        | None
    ) = None

    target_system_prompt_sha256: (
        str
        | None
    ) = None

    specialist_max_new_tokens: (
        int
        | None
    ) = None


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
    Convert verified preference records into target-specific
    specialist DPO artifacts.

    Phase-4C.2 policy:

        only exact provenance-verified specialist evidence may
        enter a newly materialized training artifact.

    Historical evidence is never silently rebound to current:

        model bytes
        tokenizer
        model profile
        agent definition
        capability catalog
        worker prompt
        user request
        routing context
        message shape
        generation contract

    Legacy evidence remains readable upstream but fails closed here.
    """

    def __init__(
        self,
        *,
        agent_definition_path: Path,
        model_profile: ModelProfileSettings,
        output_root: Path,
    ) -> None:

        self.agent_definition_path = (
            agent_definition_path
            .expanduser()
            .resolve()
        )

        self.model_profile = (
            model_profile
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

        rejected_calls = (
            record.rejected.tool_calls
        )

        chosen_calls = (
            record.chosen.tool_calls
        )

        # ====================================================
        # EXACT WORKER CALL-SET EVIDENCE
        #
        # tool_calls is a distinct response representation.
        #
        # It is used for reviewed invalid-cardinality behavior
        # such as a worker returning two calls when the runtime
        # contract allows exactly one.
        # ====================================================

        if (
            rejected_calls is not None
            or chosen_calls is not None
        ):

            if (
                rejected_calls
                != chosen_calls
            ):

                dimensions.add(
                    "tool_calls"
                )

            return (
                dimensions
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

        return (
            dimensions
        )

    def _classify(
        self,
        record: PreferenceDatasetRecord,
    ) -> tuple[
        str | None,
        str | None,
    ]:

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

        return (
            next(
                iter(
                    specialist_dimensions
                )
            ),
            None,
        )

    # ========================================================
    # TARGET VALIDATION
    # ========================================================

    def _validate_specialist_target(
        self,
        record: PreferenceDatasetRecord,
    ) -> (
        str
        | None
    ):

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

        rejected_calls = (
            record.rejected.tool_calls
        )

        chosen_calls = (
            record.chosen.tool_calls
        )

        # ====================================================
        # INVALID-CARDINALITY WORKER RESPONSE EVIDENCE
        #
        # The runtime contract permits exactly one call.
        #
        # Therefore:
        #
        # chosen:
        #     MUST contain exactly one valid target capability.
        #
        # rejected:
        #     MUST represent invalid cardinality.
        #
        # Rejected calls may contain hallucinated capability
        # names because hallucination is precisely the behavior
        # being preserved as negative evidence.
        # ====================================================

        if (
            rejected_calls is not None
            or chosen_calls is not None
        ):

            if (
                rejected_calls is None
                or chosen_calls is None
            ):

                return (
                    "mixed_worker_response_representation"
                )

            if not isinstance(
                chosen_calls,
                list,
            ):

                return (
                    "chosen_tool_calls_invalid"
                )

            if (
                len(
                    chosen_calls
                )
                != 1
            ):

                return (
                    "chosen_tool_call_count_invalid"
                )

            if not isinstance(
                rejected_calls,
                list,
            ):

                return (
                    "rejected_tool_calls_invalid"
                )

            if (
                len(
                    rejected_calls
                )
                == 1
            ):

                return (
                    "rejected_tool_call_count_not_invalid"
                )

            # --------------------------------------------
            # Chosen response must be executable by the
            # target specialist.
            # --------------------------------------------

            chosen_call = (
                chosen_calls[
                    0
                ]
            )

            if not isinstance(
                chosen_call,
                dict,
            ):

                return (
                    "chosen_tool_call_invalid"
                )

            chosen_name = (
                chosen_call.get(
                    "name"
                )
            )

            chosen_arguments = (
                chosen_call.get(
                    "arguments"
                )
            )

            if (
                not isinstance(
                    chosen_name,
                    str,
                )
                or not chosen_name
            ):

                return (
                    "chosen_tool_call_invalid"
                )

            if not isinstance(
                chosen_arguments,
                dict,
            ):

                return (
                    "chosen_tool_call_invalid"
                )

            if (
                chosen_name
                not in self.agent.tools
            ):

                return (
                    "chosen_tool_not_allowed"
                )

            # --------------------------------------------
            # Rejected response must still be structurally
            # valid worker-call evidence.
            #
            # Capability names are intentionally NOT checked
            # against the target capability list.
            #
            # A hallucinated rejected capability is legitimate
            # negative evidence.
            # --------------------------------------------

            for call in rejected_calls:

                if not isinstance(
                    call,
                    dict,
                ):

                    return (
                        "rejected_tool_call_invalid"
                    )

                name = (
                    call.get(
                        "name"
                    )
                )

                arguments = (
                    call.get(
                        "arguments"
                    )
                )

                if (
                    not isinstance(
                        name,
                        str,
                    )
                    or not name
                    or not isinstance(
                        arguments,
                        dict,
                    )
                ):

                    return (
                        "rejected_tool_call_invalid"
                    )

            return None

        # ====================================================
        # NORMAL EXACTLY-ONE-CALL PREFERENCE
        # ====================================================

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

        # Chosen behavior must be an actual target capability.
        if (
            record.chosen.tool
            not in self.agent.tools
        ):

            return (
                "chosen_tool_not_allowed"
            )

        # IMPORTANT:
        #
        # rejected.tool is deliberately NOT required to belong
        # to self.agent.tools.
        #
        # A hallucinated rejected tool is valid reviewed negative
        # evidence and must survive materialization.

        return None

    # ========================================================
    # RECORD BUILDING
    # ========================================================

    def _build_record(
        self,
        *,
        source: PreferenceDatasetRecord,
        change_type: str,
        prompt_messages: list[
            dict[
                str,
                str,
            ]
        ],
    ) -> SpecialistDpoRecord:

        if (
            source.execution_provenance
            is None
        ):

            raise ValueError(
                "Internal provenance gate error: "
                "accepted source has no execution provenance."
            )

        if (
            change_type
            == "tool_calls"
        ):

            chosen_calls = (
                source.chosen.tool_calls
            )

            rejected_calls = (
                source.rejected.tool_calls
            )

            if (
                chosen_calls is None
                or rejected_calls is None
            ):

                raise ValueError(
                    "Internal tool_calls materialization "
                    "error."
                )

            chosen = (
                _tool_calls_response(
                    chosen_calls
                )
            )

            rejected = (
                _tool_calls_response(
                    rejected_calls
                )
            )

        else:

            if (
                source.chosen.tool
                is None
                or source.rejected.tool
                is None
            ):

                raise ValueError(
                    "Internal single-call materialization "
                    "error."
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

        provenance_payload = (
            source
            .execution_provenance
            .model_dump(
                mode="json",
                by_alias=True,
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

            "source_execution_provenance":
                provenance_payload,
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

                source_execution_provenance=(
                    source
                    .execution_provenance
                    .model_copy(
                        deep=True
                    )
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

        environment: (
            SpecialistTrainingEnvironment
            | None
        ) = None

        for record in records:

            target_error = (
                self._validate_specialist_target(
                    record
                )
            )

            if (
                target_error
                is not None
            ):

                _increment_exclusion(
                    exclusion_counts,
                    target_error,
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

            if (
                exclusion_reason
                is not None
            ):

                _increment_exclusion(
                    exclusion_counts,
                    exclusion_reason,
                )

                continue

            if (
                change_type
                is None
            ):

                raise ValueError(
                    "Internal DPO classification error."
                )

            # Build the CURRENT environment only when at least one
            # record is actually a specialist-training candidate.
            #
            # This hashes the current checkpoint once for the
            # entire materialization build.
            if (
                environment
                is None
            ):

                environment = (
                    build_specialist_training_environment(
                        agent=(
                            self.agent
                        ),

                        model_profile=(
                            self.model_profile
                        ),
                    )
                )

            provenance_check = (
                validate_specialist_training_provenance(
                    environment=(
                        environment
                    ),

                    record=(
                        record
                    ),
                )
            )

            if not (
                provenance_check.accepted
            ):

                reason = (
                    provenance_check.exclusion_reason
                    or "execution_provenance_rejected"
                )

                _increment_exclusion(
                    exclusion_counts,
                    reason,
                )

                continue

            materialized.append(
                self._build_record(
                    source=(
                        record
                    ),

                    change_type=(
                        change_type
                    ),

                    prompt_messages=(
                        provenance_check
                        .prompt_messages
                    ),
                )
            )

        if not materialized:

            details = (
                ", ".join(
                    (
                        f"{reason}={count}"
                    )

                    for (
                        reason,
                        count,
                    )
                    in sorted(
                        exclusion_counts.items()
                    )
                )
            )

            suffix = (
                f" Exclusions: {details}"
                if details
                else ""
            )

            raise ValueError(
                "No provenance-verified "
                "target-compatible specialist DPO "
                "records are available."
                f"{suffix}"
            )

        if (
            environment
            is None
        ):

            raise ValueError(
                "Internal provenance environment error."
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

        if (
            target_directory.exists()
        ):

            raise ValueError(
                "Target-specific DPO artifact "
                "already exists: "
                f"{target_directory}"
            )

        target_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        fingerprint = (
            environment.model_fingerprint
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

                execution_provenance_enforced=True,

                target_model_artifact_sha256=(
                    fingerprint
                    .content_sha256
                ),

                target_model_weights_sha256=(
                    fingerprint
                    .weights_sha256
                ),

                target_tokenizer_artifact_sha256=(
                    fingerprint
                    .tokenizer_sha256
                ),

                target_model_profile_sha256=(
                    environment
                    .model_profile_sha256
                ),

                target_agent_definition_sha256=(
                    environment
                    .agent_definition_sha256
                ),

                target_capability_catalog_sha256=(
                    environment
                    .capability_catalog_sha256
                ),

                target_system_prompt_sha256=(
                    environment
                    .system_prompt_sha256
                ),

                specialist_max_new_tokens=(
                    SPECIALIST_TRAINING_MAX_NEW_TOKENS
                ),
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