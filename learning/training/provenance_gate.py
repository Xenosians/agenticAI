from __future__ import annotations

from dataclasses import (
    dataclass,
)

from typing import (
    Any,
)

from config import (
    ModelProfileSettings,
)

from learning.evidence.execution_provenance import (
    PROVENANCE_SCHEMA,
    RuntimeModelArtifactFingerprint,
    SpecialistExecutionProvenance,
    canonical_json,
    fingerprint_runtime_model_artifact,
    sha256_text,
)

from learning.evidence.types import (
    PreferenceDatasetRecord,
)

from subagents.core.definitions.types import (
    AgentDefinition,
)

from subagents.core.tooling.capabilities import (
    build_agent_capability_catalog,
)

from subagents.core.tooling.prompt import (
    build_worker_system_prompt,
)


# Must remain equal to the specialist runtime generation contract.
#
# Phase 4C.2-C will make the downstream trainer verify this again
# from the materialized manifest before any optimizer is created.
SPECIALIST_TRAINING_MAX_NEW_TOKENS = 256


# ============================================================
# ENVIRONMENT SNAPSHOT
# ============================================================


@dataclass(
    frozen=True
)
class SpecialistTrainingEnvironment:
    agent: AgentDefinition

    model_profile: ModelProfileSettings

    capability_catalog: list[
        dict[
            str,
            Any,
        ]
    ]

    system_prompt: str

    model_fingerprint: (
        RuntimeModelArtifactFingerprint
    )

    model_profile_sha256: str

    agent_definition_sha256: str

    capability_catalog_sha256: str

    system_prompt_sha256: str


@dataclass(
    frozen=True
)
class SpecialistTrainingProvenanceCheck:
    accepted: bool

    exclusion_reason: (
        str
        | None
    )

    prompt_messages: list[
        dict[
            str,
            str,
        ]
    ]

    current_provenance: (
        SpecialistExecutionProvenance
        | None
    )


# ============================================================
# IDENTITY PAYLOADS
# ============================================================


def _model_profile_identity_payload(
    profile: ModelProfileSettings,
) -> dict[
    str,
    Any,
]:
    """
    This intentionally mirrors runtime provenance construction.

    Absolute filesystem paths are not identity.

    The checkpoint itself is identified separately by
    cryptographic artifact hashes.
    """

    return {
        "backend":
            profile.backend,

        "quantization":
            profile.quantization,

        "compute_dtype":
            profile.compute_dtype,

        "device_map":
            profile.device_map,

        "dequantize_fp8":
            profile.dequantize_fp8,

        "bnb_4bit_quant_type":
            profile.bnb_4bit_quant_type,

        "bnb_4bit_use_double_quant":
            profile.bnb_4bit_use_double_quant,
    }


def _agent_identity_payload(
    agent: AgentDefinition,
) -> dict[
    str,
    Any,
]:

    return {
        "name":
            agent.name,

        "description":
            agent.description,

        "model":
            agent.model,

        "tools":
            list(
                agent.tools
            ),

        "max_steps":
            agent.max_steps,

        "system_prompt":
            agent.system_prompt,
    }


# ============================================================
# ENVIRONMENT CONSTRUCTION
# ============================================================


def build_specialist_training_environment(
    *,
    agent: AgentDefinition,
    model_profile: ModelProfileSettings,
) -> SpecialistTrainingEnvironment:
    """
    Build one immutable CURRENT training-environment snapshot.

    Model bytes are intentionally inspected from disk with
    use_cache=False.

    Training is asking:

        "What checkpoint would I train right now?"

    That is different from runtime process-snapshot semantics:

        "What checkpoint identity did this already-loaded process
         execute?"

    The two concerns must not share stale cache assumptions.
    """

    if (
        model_profile.model_path
        is None
    ):

        raise ValueError(
            "Current specialist model profile "
            "has no model_path."
        )

    capability_catalog = (
        build_agent_capability_catalog(
            agent,
            include_arguments=True,
        )
    )

    system_prompt = (
        build_worker_system_prompt(
            agent,
            capability_catalog=(
                capability_catalog
            ),
        )
    )

    model_fingerprint = (
        fingerprint_runtime_model_artifact(
            model_profile.model_path,
            use_cache=False,
        )
    )

    model_profile_sha256 = (
        sha256_text(
            canonical_json(
                _model_profile_identity_payload(
                    model_profile
                )
            )
        )
    )

    agent_definition_sha256 = (
        sha256_text(
            canonical_json(
                _agent_identity_payload(
                    agent
                )
            )
        )
    )

    capability_catalog_sha256 = (
        sha256_text(
            canonical_json(
                capability_catalog
            )
        )
    )

    system_prompt_sha256 = (
        sha256_text(
            system_prompt
        )
    )

    required_identity = [
        model_fingerprint.content_sha256,
        model_fingerprint.weights_sha256,
        model_fingerprint.tokenizer_sha256,
        model_fingerprint.config_sha256,
        model_fingerprint.chat_template_sha256,
        model_profile_sha256,
        agent_definition_sha256,
        capability_catalog_sha256,
        system_prompt_sha256,
    ]

    if not all(
        bool(
            value
        )

        for value
        in required_identity
    ):

        raise ValueError(
            "Current specialist training environment "
            "does not have complete cryptographic identity."
        )

    return (
        SpecialistTrainingEnvironment(
            agent=(
                agent
            ),

            model_profile=(
                model_profile
            ),

            capability_catalog=(
                capability_catalog
            ),

            system_prompt=(
                system_prompt
            ),

            model_fingerprint=(
                model_fingerprint
            ),

            model_profile_sha256=(
                model_profile_sha256
            ),

            agent_definition_sha256=(
                agent_definition_sha256
            ),

            capability_catalog_sha256=(
                capability_catalog_sha256
            ),

            system_prompt_sha256=(
                system_prompt_sha256
            ),
        )
    )


# ============================================================
# MESSAGE RECONSTRUCTION
# ============================================================


def _normalized_task_instructions(
    record: PreferenceDatasetRecord,
) -> (
    str
    | None
):

    if (
        record.task_instructions
        is None
    ):

        return None

    normalized = (
        record
        .task_instructions
        .strip()
    )

    if not normalized:

        return None

    return normalized


def build_current_specialist_messages(
    *,
    environment: SpecialistTrainingEnvironment,
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
                environment.system_prompt,
        },

        {
            "role":
                "user",

            "content":
                record.user_request,
        },
    ]

    normalized_instructions = (
        _normalized_task_instructions(
            record
        )
    )

    if (
        normalized_instructions
        is not None
    ):

        messages.append(
            {
                "role":
                    "user",

                "content":
                    (
                        "Additional task context "
                        "from the routing stage:\n"
                        f"{normalized_instructions}"
                    ),
            }
        )

    return messages


# ============================================================
# CURRENT PROVENANCE RECONSTRUCTION
# ============================================================


def reconstruct_current_training_provenance(
    *,
    environment: SpecialistTrainingEnvironment,
    record: PreferenceDatasetRecord,
) -> tuple[
    SpecialistExecutionProvenance,
    list[
        dict[
            str,
            str,
        ]
    ],
]:

    messages = (
        build_current_specialist_messages(
            environment=(
                environment
            ),

            record=(
                record
            ),
        )
    )

    normalized_instructions = (
        _normalized_task_instructions(
            record
        )
    )

    fingerprint = (
        environment.model_fingerprint
    )

    provenance = (
        SpecialistExecutionProvenance(
            provenance_complete=True,

            agent_name=(
                environment.agent.name
            ),

            model_key=(
                environment.agent.model
            ),

            backend=(
                environment
                .model_profile
                .backend
            ),

            quantization=(
                environment
                .model_profile
                .quantization
            ),

            compute_dtype=(
                environment
                .model_profile
                .compute_dtype
            ),

            device_map=(
                environment
                .model_profile
                .device_map
            ),

            bnb_4bit_quant_type=(
                environment
                .model_profile
                .bnb_4bit_quant_type
            ),

            bnb_4bit_use_double_quant=(
                environment
                .model_profile
                .bnb_4bit_use_double_quant
            ),

            model_profile_sha256=(
                environment
                .model_profile_sha256
            ),

            model_artifact_sha256=(
                fingerprint
                .content_sha256
            ),

            model_weights_sha256=(
                fingerprint
                .weights_sha256
            ),

            model_config_sha256=(
                fingerprint
                .config_sha256
            ),

            generation_config_sha256=(
                fingerprint
                .generation_config_sha256
            ),

            tokenizer_artifact_sha256=(
                fingerprint
                .tokenizer_sha256
            ),

            tokenizer_config_sha256=(
                fingerprint
                .tokenizer_config_sha256
            ),

            tokenizer_json_sha256=(
                fingerprint
                .tokenizer_json_sha256
            ),

            chat_template_sha256=(
                fingerprint
                .chat_template_sha256
            ),

            agent_definition_sha256=(
                environment
                .agent_definition_sha256
            ),

            capability_catalog_sha256=(
                environment
                .capability_catalog_sha256
            ),

            system_prompt_sha256=(
                environment
                .system_prompt_sha256
            ),

            messages_sha256=(
                sha256_text(
                    canonical_json(
                        messages
                    )
                )
            ),

            user_request_sha256=(
                sha256_text(
                    record.user_request
                )
            ),

            task_instructions_sha256=(
                sha256_text(
                    normalized_instructions
                )

                if (
                    normalized_instructions
                    is not None
                )

                else None
            ),

            max_new_tokens=(
                SPECIALIST_TRAINING_MAX_NEW_TOKENS
            ),
        )
    )

    return (
        provenance,
        messages,
    )


# ============================================================
# MISMATCH CLASSIFICATION
# ============================================================


PROVENANCE_COMPARISON_GROUPS: tuple[
    tuple[
        str,
        tuple[
            str,
            ...,
        ],
    ],
    ...,
] = (
    (
        "runtime_profile",
        (
            "backend",
            "quantization",
            "compute_dtype",
            "device_map",
            "bnb_4bit_quant_type",
            "bnb_4bit_use_double_quant",
            "model_profile_sha256",
        ),
    ),

    (
        "model_checkpoint",
        (
            "model_artifact_sha256",
            "model_weights_sha256",
            "model_config_sha256",
            "generation_config_sha256",
        ),
    ),

    (
        "tokenizer",
        (
            "tokenizer_artifact_sha256",
            "tokenizer_config_sha256",
            "tokenizer_json_sha256",
            "chat_template_sha256",
        ),
    ),

    (
        "agent_definition",
        (
            "agent_definition_sha256",
        ),
    ),

    (
        "capability_catalog",
        (
            "capability_catalog_sha256",
        ),
    ),

    (
        "system_prompt",
        (
            "system_prompt_sha256",
        ),
    ),

    (
        "request_context",
        (
            "user_request_sha256",
            "task_instructions_sha256",
        ),
    ),

    (
        "messages",
        (
            "messages_sha256",
        ),
    ),

    (
        "generation_contract",
        (
            "max_new_tokens",
        ),
    ),
)


def _first_provenance_mismatch(
    *,
    historical: SpecialistExecutionProvenance,
    current: SpecialistExecutionProvenance,
) -> (
    str
    | None
):

    for (
        group,
        fields,
    ) in PROVENANCE_COMPARISON_GROUPS:

        for field in fields:

            if (
                getattr(
                    historical,
                    field,
                )
                != getattr(
                    current,
                    field,
                )
            ):

                return (
                    "execution_provenance_"
                    f"{group}_mismatch"
                )

    return None


# ============================================================
# FAIL-CLOSED CHECK
# ============================================================


def validate_specialist_training_provenance(
    *,
    environment: SpecialistTrainingEnvironment,
    record: PreferenceDatasetRecord,
) -> SpecialistTrainingProvenanceCheck:
    """
    Validate historical evidence against the exact CURRENT
    specialist training environment.

    Legacy or mismatched evidence is excluded.

    Nothing here attempts to "repair" old evidence by binding it to
    present-day configuration.
    """

    historical = (
        record.execution_provenance
    )

    if historical is None:

        return (
            SpecialistTrainingProvenanceCheck(
                accepted=False,

                exclusion_reason=(
                    "missing_execution_provenance"
                ),

                prompt_messages=[],

                current_provenance=None,
            )
        )

    if (
        historical.schema_name
        != PROVENANCE_SCHEMA
    ):

        return (
            SpecialistTrainingProvenanceCheck(
                accepted=False,

                exclusion_reason=(
                    "unsupported_execution_provenance_schema"
                ),

                prompt_messages=[],

                current_provenance=None,
            )
        )

    if not (
        historical.provenance_complete
    ):

        return (
            SpecialistTrainingProvenanceCheck(
                accepted=False,

                exclusion_reason=(
                    "incomplete_execution_provenance"
                ),

                prompt_messages=[],

                current_provenance=None,
            )
        )

    if (
        historical.agent_name
        != environment.agent.name
        or historical.model_key
        != environment.agent.model
    ):

        return (
            SpecialistTrainingProvenanceCheck(
                accepted=False,

                exclusion_reason=(
                    "execution_provenance_target_mismatch"
                ),

                prompt_messages=[],

                current_provenance=None,
            )
        )

    (
        current,
        messages,
    ) = (
        reconstruct_current_training_provenance(
            environment=(
                environment
            ),

            record=(
                record
            ),
        )
    )

    mismatch = (
        _first_provenance_mismatch(
            historical=(
                historical
            ),

            current=(
                current
            ),
        )
    )

    if (
        mismatch
        is not None
    ):

        return (
            SpecialistTrainingProvenanceCheck(
                accepted=False,

                exclusion_reason=(
                    mismatch
                ),

                prompt_messages=(
                    messages
                ),

                current_provenance=(
                    current
                ),
            )
        )

    return (
        SpecialistTrainingProvenanceCheck(
            accepted=True,

            exclusion_reason=None,

            prompt_messages=(
                messages
            ),

            current_provenance=(
                current
            ),
        )
    )