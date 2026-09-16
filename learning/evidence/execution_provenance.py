from __future__ import annotations

import hashlib
import json
import threading

from pathlib import (
    Path,
)

from typing import (
    Any,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from config import (
    ModelProfileSettings,
)

from subagents.core.types import (
    AgentDefinition,
)


# ============================================================
# CONSTANTS
# ============================================================


PROVENANCE_SCHEMA = (
    "specialist-execution-provenance.v1"
)


RUNTIME_MODEL_EXPLICIT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
    "vocab.txt",
    "tokenizer.model",
    "spiece.model",
)


RUNTIME_MODEL_PATTERNS = (
    "*.safetensors",
    "pytorch_model*.bin",
    "*.index.json",
)


TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
    "vocab.txt",
    "tokenizer.model",
    "spiece.model",
}


# ============================================================
# HASH HELPERS
# ============================================================


def canonical_json(
    value: Any,
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


def sha256_text(
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


def sha256_file(
    path: Path,
) -> str:

    digest = (
        hashlib.sha256()
    )

    with path.open(
        "rb"
    ) as handle:

        while True:

            chunk = (
                handle.read(
                    1024
                    * 1024
                )
            )

            if not chunk:

                break

            digest.update(
                chunk
            )

    return (
        digest.hexdigest()
    )


# ============================================================
# MODEL ARTIFACT FINGERPRINT
# ============================================================


class RuntimeModelArtifactFile(
    BaseModel
):
    path: str

    size_bytes: int

    sha256: str


class RuntimeModelArtifactFingerprint(
    BaseModel
):
    model_config = (
        ConfigDict(
            extra="forbid",
        )
    )

    content_sha256: str

    weights_sha256: str

    tokenizer_sha256: str

    config_sha256: str

    generation_config_sha256: (
        str
        | None
    ) = None

    tokenizer_config_sha256: (
        str
        | None
    ) = None

    tokenizer_json_sha256: (
        str
        | None
    ) = None

    chat_template_sha256: (
        str
        | None
    ) = None

    files: list[
        RuntimeModelArtifactFile
    ] = Field(
        default_factory=list
    )


# ============================================================
# PROCESS-SNAPSHOT CACHE
# ============================================================


_MODEL_FINGERPRINT_CACHE: dict[
    str,
    RuntimeModelArtifactFingerprint,
] = {}


_MODEL_FINGERPRINT_LOCK = (
    threading.Lock()
)


# ============================================================
# MODEL FILE DISCOVERY
# ============================================================


def _collect_runtime_model_files(
    model_path: Path,
) -> list[
    Path
]:

    resolved = (
        model_path
        .expanduser()
        .resolve()
    )

    if not resolved.is_dir():

        raise ValueError(
            "Runtime model directory does not exist: "
            f"{resolved}"
        )

    candidate_paths: set[
        Path
    ] = set()

    for name in (
        RUNTIME_MODEL_EXPLICIT_FILES
    ):

        path = (
            resolved
            / name
        )

        if path.is_file():

            candidate_paths.add(
                path
            )

    for pattern in (
        RUNTIME_MODEL_PATTERNS
    ):

        for path in (
            resolved.glob(
                pattern
            )
        ):

            if path.is_file():

                candidate_paths.add(
                    path
                )

    return (
        sorted(
            candidate_paths,

            key=lambda item: (
                str(
                    item.relative_to(
                        resolved
                    )
                )
            ),
        )
    )


# ============================================================
# MODEL FINGERPRINT HELPERS
# ============================================================


def _aggregate_files(
    files: list[
        RuntimeModelArtifactFile
    ],
) -> str:

    payload = [
        item.model_dump(
            mode="json"
        )

        for item
        in files
    ]

    return (
        sha256_text(
            canonical_json(
                payload
            )
        )
    )


def _find_file_hash(
    files: list[
        RuntimeModelArtifactFile
    ],
    name: str,
) -> (
    str
    | None
):

    for item in files:

        if (
            item.path
            == name
        ):

            return (
                item.sha256
            )

    return None


def _derive_chat_template_sha256(
    *,
    model_path: Path,
    files: list[
        RuntimeModelArtifactFile
    ],
) -> (
    str
    | None
):

    explicit_template_hash = (
        _find_file_hash(
            files,
            "chat_template.jinja",
        )
    )

    if (
        explicit_template_hash
        is not None
    ):

        return (
            explicit_template_hash
        )

    tokenizer_config_path = (
        model_path
        / "tokenizer_config.json"
    )

    if not (
        tokenizer_config_path
        .is_file()
    ):

        return None

    try:

        payload = (
            json.loads(
                tokenizer_config_path
                .read_text(
                    encoding="utf-8"
                )
            )
        )

    except Exception:

        return None

    if not isinstance(
        payload,
        dict,
    ):

        return None

    chat_template = (
        payload.get(
            "chat_template"
        )
    )

    if (
        chat_template
        is None
    ):

        return None

    if isinstance(
        chat_template,
        str,
    ):

        return (
            sha256_text(
                chat_template
            )
        )

    return (
        sha256_text(
            canonical_json(
                chat_template
            )
        )
    )


# ============================================================
# MODEL ARTIFACT FINGERPRINTING
# ============================================================


def fingerprint_runtime_model_artifact(
    model_path: Path,
    *,
    use_cache: bool = True,
) -> RuntimeModelArtifactFingerprint:
    """
    Cryptographically identify the local checkpoint used by the
    specialist runtime.

    Identity includes:

        model weights
        model configuration
        generation configuration
        tokenizer artifacts
        chat-template identity

    Cache semantics intentionally represent a process-local model
    deployment snapshot.

    Once a model path is fingerprinted, repeated cached calls
    return the same identity even if files on disk later change.

    This matches runtime semantics:

        modifying checkpoint files on disk does not mutate a model
        that has already been loaded into memory.

    Code that intentionally needs to inspect current disk contents
    must either:

        fingerprint_runtime_model_artifact(
            path,
            use_cache=False,
        )

    or invalidate the snapshot with:

        clear_runtime_model_fingerprint_cache(path)

    Filesystem timestamps are deliberately NOT treated as proof of
    artifact identity.
    """

    resolved = (
        model_path
        .expanduser()
        .resolve()
    )

    cache_key = (
        str(
            resolved
        )
    )

    # ========================================================
    # PROCESS SNAPSHOT CACHE
    # ========================================================

    if use_cache:

        with (
            _MODEL_FINGERPRINT_LOCK
        ):

            cached = (
                _MODEL_FINGERPRINT_CACHE
                .get(
                    cache_key
                )
            )

            if (
                cached
                is not None
            ):

                return (
                    cached
                )

    # ========================================================
    # DISCOVER MODEL ARTIFACTS
    # ========================================================

    paths = (
        _collect_runtime_model_files(
            resolved
        )
    )

    if not paths:

        raise ValueError(
            "Runtime model directory contains no "
            "recognized model artifacts."
        )

    config_path = (
        resolved
        / "config.json"
    )

    if not (
        config_path
        .is_file()
    ):

        raise ValueError(
            "Runtime model is missing config.json."
        )

    # ========================================================
    # WEIGHT FILES
    # ========================================================

    weight_paths = [
        path

        for path
        in paths

        if (
            path.suffix
            == ".safetensors"

            or (
                path.suffix
                == ".bin"

                and path.name.startswith(
                    "pytorch_model"
                )
            )
        )
    ]

    if not weight_paths:

        raise ValueError(
            "Runtime model contains no supported "
            "weight files."
        )

    # ========================================================
    # TOKENIZER FILES
    # ========================================================

    tokenizer_paths = [
        path

        for path
        in paths

        if (
            str(
                path.relative_to(
                    resolved
                )
            )
            in TOKENIZER_FILES
        )
    ]

    if not tokenizer_paths:

        raise ValueError(
            "Runtime model contains no recognized "
            "tokenizer artifacts."
        )

    # ========================================================
    # CRYPTOGRAPHIC FILE HASHING
    # ========================================================

    files = [
        RuntimeModelArtifactFile(
            path=(
                str(
                    path.relative_to(
                        resolved
                    )
                )
            ),

            size_bytes=(
                path.stat().st_size
            ),

            sha256=(
                sha256_file(
                    path
                )
            ),
        )

        for path
        in paths
    ]

    # ========================================================
    # GROUP MEMBERSHIP
    # ========================================================

    weight_names = {
        str(
            path.relative_to(
                resolved
            )
        )

        for path
        in weight_paths
    }

    tokenizer_names = {
        str(
            path.relative_to(
                resolved
            )
        )

        for path
        in tokenizer_paths
    }

    weight_files = [
        item

        for item
        in files

        if (
            item.path
            in weight_names
        )
    ]

    tokenizer_files = [
        item

        for item
        in files

        if (
            item.path
            in tokenizer_names
        )
    ]

    # ========================================================
    # BUILD FINGERPRINT
    # ========================================================

    result = (
        RuntimeModelArtifactFingerprint(
            content_sha256=(
                _aggregate_files(
                    files
                )
            ),

            weights_sha256=(
                _aggregate_files(
                    weight_files
                )
            ),

            tokenizer_sha256=(
                _aggregate_files(
                    tokenizer_files
                )
            ),

            config_sha256=(
                _find_file_hash(
                    files,
                    "config.json",
                )
                or ""
            ),

            generation_config_sha256=(
                _find_file_hash(
                    files,
                    "generation_config.json",
                )
            ),

            tokenizer_config_sha256=(
                _find_file_hash(
                    files,
                    "tokenizer_config.json",
                )
            ),

            tokenizer_json_sha256=(
                _find_file_hash(
                    files,
                    "tokenizer.json",
                )
            ),

            chat_template_sha256=(
                _derive_chat_template_sha256(
                    model_path=(
                        resolved
                    ),

                    files=(
                        files
                    ),
                )
            ),

            files=(
                files
            ),
        )
    )

    if not (
        result.config_sha256
    ):

        raise ValueError(
            "Unable to fingerprint model config.json."
        )

    # ========================================================
    # CACHE PROCESS SNAPSHOT
    # ========================================================

    if use_cache:

        with (
            _MODEL_FINGERPRINT_LOCK
        ):

            _MODEL_FINGERPRINT_CACHE[
                cache_key
            ] = (
                result
            )

    return (
        result
    )


def clear_runtime_model_fingerprint_cache(
    model_path: (
        Path
        | None
    ) = None,
) -> None:
    """
    Invalidate process-local model deployment identity.

    No argument:

        clear every cached model fingerprint

    Path supplied:

        clear only that model's cached fingerprint

    Future intentional ModelManager reload/promotion paths should
    invalidate the corresponding fingerprint before constructing
    a fresh backend from changed checkpoint contents.
    """

    with (
        _MODEL_FINGERPRINT_LOCK
    ):

        if (
            model_path
            is None
        ):

            _MODEL_FINGERPRINT_CACHE.clear()

            return

        resolved = (
            model_path
            .expanduser()
            .resolve()
        )

        _MODEL_FINGERPRINT_CACHE.pop(
            str(
                resolved
            ),
            None,
        )


# ============================================================
# SPECIALIST EXECUTION PROVENANCE
# ============================================================


class SpecialistExecutionProvenance(
    BaseModel
):
    """
    Immutable identity evidence for one specialist model
    execution.

    Raw prompts and capability metadata are not duplicated into
    the trajectory. Instead their exact cryptographic identities
    are captured.

    Historical trajectories created before this schema may omit
    provenance entirely.

    Such legacy evidence must later fail closed for production
    training.
    """

    model_config = (
        ConfigDict(
            populate_by_name=True,
            extra="forbid",
        )
    )

    schema_name: str = Field(
        default=(
            PROVENANCE_SCHEMA
        ),
        alias="schema",
    )

    provenance_complete: bool

    # ========================================================
    # SPECIALIST IDENTITY
    # ========================================================

    agent_name: str

    model_key: str

    # ========================================================
    # RUNTIME MODEL PROFILE
    # ========================================================

    backend: str

    quantization: str

    compute_dtype: str

    device_map: (
        str
        | None
    ) = None

    bnb_4bit_quant_type: str

    bnb_4bit_use_double_quant: bool

    model_profile_sha256: str

    # ========================================================
    # MODEL CHECKPOINT IDENTITY
    # ========================================================

    model_artifact_sha256: str

    model_weights_sha256: str

    model_config_sha256: str

    generation_config_sha256: (
        str
        | None
    ) = None

    # ========================================================
    # TOKENIZER / TEMPLATE IDENTITY
    # ========================================================

    tokenizer_artifact_sha256: str

    tokenizer_config_sha256: (
        str
        | None
    ) = None

    tokenizer_json_sha256: (
        str
        | None
    ) = None

    chat_template_sha256: (
        str
        | None
    ) = None

    # ========================================================
    # PROMPT / CAPABILITY IDENTITY
    # ========================================================

    agent_definition_sha256: str

    capability_catalog_sha256: str

    system_prompt_sha256: str

    messages_sha256: str

    # ========================================================
    # REQUEST CONTEXT
    # ========================================================

    user_request_sha256: str

    task_instructions_sha256: (
        str
        | None
    ) = None

    # ========================================================
    # GENERATION CONTRACT
    # ========================================================

    max_new_tokens: int = Field(
        ge=1
    )


# ============================================================
# RUNTIME PROFILE IDENTITY
# ============================================================


def _model_profile_identity_payload(
    profile: ModelProfileSettings,
) -> dict[
    str,
    Any,
]:
    """
    Describe model runtime behavior without binding identity to an
    absolute filesystem location.

    Checkpoint contents are identified independently through
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


# ============================================================
# AGENT DEFINITION IDENTITY
# ============================================================


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
# PROVENANCE CONSTRUCTION
# ============================================================


def build_specialist_execution_provenance(
    *,
    agent: AgentDefinition,
    model_profile: ModelProfileSettings,
    capability_catalog: list[
        dict[
            str,
            Any,
        ]
    ],
    messages: list[
        dict[
            str,
            str,
        ]
    ],
    user_request: str,
    task_instructions: (
        str
        | None
    ),
    max_new_tokens: int,
) -> SpecialistExecutionProvenance:
    """
    Build immutable execution identity for one specialist
    invocation.

    `capability_catalog` must be the exact same catalog object used
    to construct the system prompt contained in `messages`.

    AgentRuntime enforces that invariant by resolving capabilities
    once and reusing the result for both prompt construction and
    provenance capture.
    """

    # ========================================================
    # BASIC CONTRACT VALIDATION
    # ========================================================

    if not (
        agent.model
        .strip()
    ):

        raise ValueError(
            "Agent model key is empty."
        )

    if (
        model_profile.model_path
        is None
    ):

        raise ValueError(
            "Specialist model profile has no local "
            "model_path."
        )

    if not messages:

        raise ValueError(
            "Specialist execution messages are empty."
        )

    first_message = (
        messages[
            0
        ]
    )

    if (
        first_message.get(
            "role"
        )
        != "system"
    ):

        raise ValueError(
            "Specialist execution must begin with "
            "a system message."
        )

    system_prompt = (
        first_message.get(
            "content"
        )
    )

    if (
        not isinstance(
            system_prompt,
            str,
        )
        or not system_prompt
    ):

        raise ValueError(
            "Specialist system prompt is empty."
        )

    if (
        max_new_tokens
        < 1
    ):

        raise ValueError(
            "max_new_tokens must be positive."
        )

    # ========================================================
    # MODEL ARTIFACT IDENTITY
    # ========================================================

    model_fingerprint = (
        fingerprint_runtime_model_artifact(
            model_profile.model_path
        )
    )

    # ========================================================
    # MODEL PROFILE IDENTITY
    # ========================================================

    profile_payload = (
        _model_profile_identity_payload(
            model_profile
        )
    )

    # ========================================================
    # AGENT IDENTITY
    # ========================================================

    agent_payload = (
        _agent_identity_payload(
            agent
        )
    )

    # ========================================================
    # TASK CONTEXT NORMALIZATION
    #
    # AgentRuntime passes the normalized value used to construct
    # the actual messages. The fallback here keeps direct callers
    # deterministic.
    # ========================================================

    normalized_task_instructions = (
        task_instructions.strip()

        if (
            task_instructions
            is not None
            and task_instructions.strip()
        )

        else None
    )

    # ========================================================
    # COMPLETENESS
    # ========================================================

    provenance_complete = (
        all(
            [
                bool(
                    model_fingerprint
                    .content_sha256
                ),

                bool(
                    model_fingerprint
                    .weights_sha256
                ),

                bool(
                    model_fingerprint
                    .tokenizer_sha256
                ),

                bool(
                    model_fingerprint
                    .config_sha256
                ),

                bool(
                    model_fingerprint
                    .chat_template_sha256
                ),

                bool(
                    system_prompt
                ),

                bool(
                    capability_catalog
                ),
            ]
        )
    )

    # ========================================================
    # RESULT
    # ========================================================

    return (
        SpecialistExecutionProvenance(
            provenance_complete=(
                provenance_complete
            ),

            agent_name=(
                agent.name
            ),

            model_key=(
                agent.model
            ),

            backend=(
                model_profile.backend
            ),

            quantization=(
                model_profile.quantization
            ),

            compute_dtype=(
                model_profile.compute_dtype
            ),

            device_map=(
                model_profile.device_map
            ),

            bnb_4bit_quant_type=(
                model_profile
                .bnb_4bit_quant_type
            ),

            bnb_4bit_use_double_quant=(
                model_profile
                .bnb_4bit_use_double_quant
            ),

            model_profile_sha256=(
                sha256_text(
                    canonical_json(
                        profile_payload
                    )
                )
            ),

            model_artifact_sha256=(
                model_fingerprint
                .content_sha256
            ),

            model_weights_sha256=(
                model_fingerprint
                .weights_sha256
            ),

            model_config_sha256=(
                model_fingerprint
                .config_sha256
            ),

            generation_config_sha256=(
                model_fingerprint
                .generation_config_sha256
            ),

            tokenizer_artifact_sha256=(
                model_fingerprint
                .tokenizer_sha256
            ),

            tokenizer_config_sha256=(
                model_fingerprint
                .tokenizer_config_sha256
            ),

            tokenizer_json_sha256=(
                model_fingerprint
                .tokenizer_json_sha256
            ),

            chat_template_sha256=(
                model_fingerprint
                .chat_template_sha256
            ),

            agent_definition_sha256=(
                sha256_text(
                    canonical_json(
                        agent_payload
                    )
                )
            ),

            capability_catalog_sha256=(
                sha256_text(
                    canonical_json(
                        capability_catalog
                    )
                )
            ),

            system_prompt_sha256=(
                sha256_text(
                    system_prompt
                )
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
                    user_request
                )
            ),

            task_instructions_sha256=(
                sha256_text(
                    normalized_task_instructions
                )

                if (
                    normalized_task_instructions
                    is not None
                )

                else None
            ),

            max_new_tokens=(
                max_new_tokens
            ),
        )
    )