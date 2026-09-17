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

from learning.training.specialist_response import (
    preference_option_calls,
    preference_option_shape,
    serialize_preference_option,
    validate_specialist_preference_pair,
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

    Phase-4D.2 specialist-response policy:

        chosen behavior
            must resolve to exactly one currently allowed
            specialist capability

        rejected behavior
            is negative evidence only

            it may contain:
                - a hallucinated tool name
                - multiple tool calls

            rejected behavior is never executed here

    This distinction is critical because malformed or hallucinated
    worker generations are precisely the behavior DPO may need to
    discourage.
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

        if (
            record.rejected.tool_calls
            != record.chosen.tool_calls
        ):

            dimensions.add(
                "tool_calls"
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
        """
        Validate that a preference record belongs to this target
        specialist and can be interpreted as specialist-response
        evidence.

        IMPORTANT:

        Rejected tools are NOT required to appear in the current
        trusted capability catalog.

        They are negative model-output evidence, not authorization.

        The chosen response remains fail-closed.
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

        rejected_has_call_set = (
            record.rejected.tool_calls
            is not None
        )

        chosen_has_call_set = (
            record.chosen.tool_calls
            is not None
        )

        # ----------------------------------------------------
        # Preserve historical singular-tool exclusion semantics.
        #
        # Old records without tool_calls still receive the same
        # missing_tool / missing_arguments reason names.
        # ----------------------------------------------------

        if (
            not rejected_has_call_set
            and not chosen_has_call_set
        ):

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

        # ----------------------------------------------------
        # Resolve representation shape.
        # ----------------------------------------------------

        try:

            rejected_shape = (
                preference_option_shape(
                    record.rejected,
                    label=(
                        "rejected"
                    ),
                )
            )

            chosen_shape = (
                preference_option_shape(
                    record.chosen,
                    label=(
                        "chosen"
                    ),
                )
            )

        except ValueError:

            return (
                "invalid_specialist_response_shape"
            )

        if (
            rejected_shape
            != chosen_shape
        ):

            return (
                "mixed_specialist_response_shape"
            )

        # ----------------------------------------------------
        # Resolve exact structured worker responses.
        # ----------------------------------------------------

        try:

            rejected_calls = (
                preference_option_calls(
                    record.rejected,
                    label=(
                        "rejected"
                    ),
                )
            )

            chosen_calls = (
                preference_option_calls(
                    record.chosen,
                    label=(
                        "chosen"
                    ),
                )
            )

        except ValueError:

            return (
                "invalid_specialist_response"
            )

        # ----------------------------------------------------
        # CHOSEN = authoritative desired specialist behavior.
        #
        # Exactly one allowed call.
        # ----------------------------------------------------

        if (
            len(
                chosen_calls
            )
            != 1
        ):

            return (
                "chosen_tool_call_count_invalid"
            )

        chosen_tool = (
            chosen_calls[
                0
            ][
                "name"
            ]
        )

        if (
            chosen_tool
            not in self.agent.tools
        ):

            return (
                "chosen_tool_not_allowed"
            )

        # ----------------------------------------------------
        # REJECTED call-set representation is reserved for a
        # genuine invalid multi-call generation.
        #
        # A one-element tool_calls negative should instead use the
        # historical singular representation.
        # ----------------------------------------------------

        if (
            rejected_shape
            == "tool_calls"
            and len(
                rejected_calls
            )
            == 1
        ):

            return (
                "rejected_tool_call_count_not_invalid"
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

        # ----------------------------------------------------
        # Serialize BOTH sides through the canonical specialist
        # worker-response codec.
        #
        # Singular behavior:
        #
        #   [{"name":"account_status","arguments":{...}}]
        #
        # Multi-call rejected behavior:
        #
        #   [
        #       {"name":"reset_password","arguments":{...}},
        #       {"name":"reset_password","arguments":{...}}
        #   ]
        #
        # No synthetic reconstruction of the rejected behavior is
        # performed here.
        # ----------------------------------------------------

        chosen = (
            serialize_preference_option(
                source.chosen,
                label=(
                    "chosen"
                ),
            )
        )

        rejected = (
            serialize_preference_option(
                source.rejected,
                label=(
                    "rejected"
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

            # =================================================
            # TARGET / RESPONSE-SHAPE VALIDATION
            # =================================================

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

            # =================================================
            # CHANGE CLASSIFICATION
            # =================================================

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

            # =================================================
            # FINAL RESPONSE-PAIR POLICY VALIDATION
            #
            # This is defense in depth.
            #
            # The dedicated codec is the authoritative definition
            # of a specialist DPO response pair.
            # =================================================

            try:

                validate_specialist_preference_pair(
                    rejected=(
                        record.rejected
                    ),

                    chosen=(
                        record.chosen
                    ),

                    allowed_tools=(
                        set(
                            self.agent.tools
                        )
                    ),
                )

            except ValueError:

                _increment_exclusion(
                    exclusion_counts,
                    "invalid_specialist_response_pair",
                )

                continue

            # =================================================
            # CURRENT TARGET ENVIRONMENT
            #
            # Build only when at least one record has crossed the
            # target/response-shape filters.
            #
            # This hashes the current checkpoint once for the
            # entire materialization build.
            # =================================================

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

            # =================================================
            # EXECUTION PROVENANCE GATE
            # =================================================

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

        # ====================================================
        # EMPTY BUILD = FAIL CLOSED
        # ====================================================

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

        # ====================================================
        # DPO RECORD-ID UNIQUENESS
        # ====================================================

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

        # ====================================================
        # CONTENT SERIALIZATION
        # ====================================================

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

        # ====================================================
        # IMMUTABLE TARGET ARTIFACT
        # ====================================================

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

        # ====================================================
        # ATOMIC-ENOUGH ARTIFACT WRITE
        # ====================================================

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
