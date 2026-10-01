from __future__ import annotations

import json
import shutil

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from config.path_portability import (
    resolve_portable_path,
)

from learning.continual.storage import (
    fingerprint_directory,
    immutable_write_json,
    immutable_write_jsonl,
    sha256_file,
)
from learning.paths import REPOSITORY_ROOT, RUNTIME_LEARNING_ROOT
from learning.training.hub_training_contracts import (
    build_hub_training_environment,
    canonical_json,
    sha256_text,
    validate_hub_response_contract,
)


DEFAULT_HUB_MATERIALIZATION_ROOT = (
    RUNTIME_LEARNING_ROOT / "phase5" / "materialized"
)

DEFAULT_MEMBERSHIP_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "continual"
    / "training-membership"
)


HUB_CHAPTER1_CURRICULUM_PROMPT_SCHEMA = (
    "hub-chapter1-structured-instructions.v1\n"
    "You are practicing the Hub's strict structured-output format.\n"
    "Return ONLY one JSON object.\n"
    "The top-level object must contain a delegations list.\n"
    "For a request that needs no specialist or tool action, return exactly:\n"
    "{\"delegations\":[]}\n"
    "Do not return prose.\n"
    "Do not use Markdown fences.\n"
    "Do not rename delegations.\n"
    "Do not use null for delegations.\n"
    "Do not invent extra top-level fields."
)


def _curriculum_prompt_for_chapter(
    chapter_id: str,
    *,
    runtime_system_prompt: str,
) -> str:
    if chapter_id == "hub-01-structured-instructions":
        return HUB_CHAPTER1_CURRICULUM_PROMPT_SCHEMA

    return runtime_system_prompt


class HubSftRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_name: str = Field(
        default="phase5-sft-record.v1",
        alias="schema",
    )

    record_id: str
    member_id: str
    partition: Literal["train", "validation"]
    role: str
    source_kind: str

    prompt_messages: list[dict[str, str]]
    chosen: str

    source_id: str
    source_artifact: str
    source_artifact_sha256: str
    lineage_id: str


class HubDpoRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_name: str = Field(
        default="phase5-dpo-record.v1",
        alias="schema",
    )

    record_id: str
    member_id: str
    partition: Literal["train", "validation"]
    role: str
    source_kind: str

    prompt_messages: list[dict[str, str]]
    chosen: str
    rejected: str

    source_id: str
    source_artifact: str
    source_artifact_sha256: str
    lineage_id: str


class HubTrainingMaterializationManifest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_name: str = Field(
        default="phase5-training-materialization.v1",
        alias="schema",
    )

    materialization_id: str
    created_at: str

    target_component: str
    target_model_key: str

    source_plan_id: str
    source_plan_directory: str
    source_plan_manifest_sha256: str
    source_plan_members_sha256: str

    curriculum_id: str
    curriculum_version: str
    chapter_id: str

    base_model_path: str
    base_model_sha256: str

    hub_system_prompt_sha256: str
    hub_capability_catalog_sha256: str
    hub_agent_definitions_sha256: str
    curriculum_prompt_sha256: str

    # Target-aware evaluation contract.
    #
    # Existing Phase-5 Hub materializations default to the original
    # strict routing-contract evaluation. External developer corpus
    # materializations use developer_sft_loss instead.
    evaluation_contract: Literal[
        "hub_contract",
        "developer_sft_loss",
    ] = "hub_contract"

    # Immutable specialist definition used when materializing a
    # non-Hub target. The trainer verifies this again immediately
    # before optimizer execution.
    target_contract_path: str | None = None
    target_contract_sha256: str | None = None

    # Token budget verified before GPU/model loading.
    #
    # Existing Hub materializations remain backwards-compatible.
    sequence_budget_tokens: int | None = None
    sequence_budget_verified: bool = False

    source_fingerprint_count: int
    source_fingerprints_verified: bool

    sft_train_count: int
    sft_validation_count: int
    dpo_train_count: int
    dpo_validation_count: int

    sft_train_sha256: str
    sft_validation_sha256: str
    dpo_train_sha256: str
    dpo_validation_sha256: str

    contract_validated_record_count: int
    excluded_counts: dict[str, int] = Field(default_factory=dict)

    lineage_isolated: bool = True
    ready_for_training: bool = False

    training_authorized: bool = False
    promotion_authorized: bool = False


class HubTrainingMaterializationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: HubTrainingMaterializationManifest
    output_directory: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required_sha256(path: Path) -> str:
    value = sha256_file(path)
    if not value:
        raise ValueError(f"Could not fingerprint file: {path}")
    return value


def _resolve_artifact_path(
    value: str,
) -> Path:
    return resolve_portable_path(
        value,
        base=REPOSITORY_ROOT,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"Invalid JSON artifact {path}: {exc}"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            value = json.loads(line)
        except Exception as exc:
            raise ValueError(
                f"Invalid JSONL at {path}:{line_number}: {exc}"
            ) from exc

        if not isinstance(value, dict):
            raise ValueError(
                f"Expected object at {path}:{line_number}"
            )

        result.append(value)

    return result


def _index_source_records(
    *,
    source_artifacts: set[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}

    identity_fields = [
        "record_id",
        "lesson_id",
        "correction_id",
        "trajectory_id",
    ]

    for artifact in sorted(source_artifacts):
        path = _resolve_artifact_path(artifact)

        if not path.is_file():
            raise ValueError(
                "Membership source artifact does not exist: "
                f"{path}"
            )

        for item in _read_jsonl(path):
            for field in identity_fields:
                value = item.get(field)

                if isinstance(value, str) and value.strip():
                    key = (artifact, value.strip())
                    prior = index.get(key)

                    if prior is not None and prior != item:
                        raise ValueError(
                            "Ambiguous source identity "
                            f"{artifact}:{value}"
                        )

                    index[key] = item

    return index


def _verify_source_fingerprints(
    manifest: dict[str, Any],
) -> dict[str, str]:
    source_fingerprints = manifest.get("source_fingerprints")

    if not isinstance(source_fingerprints, dict):
        raise ValueError(
            "Membership manifest has no source_fingerprints map."
        )

    verified: dict[str, str] = {}

    for artifact, expected in sorted(source_fingerprints.items()):
        if (
            not isinstance(artifact, str)
            or not isinstance(expected, str)
            or not expected
        ):
            raise ValueError(
                "Membership source fingerprint entry is malformed."
            )

        path = _resolve_artifact_path(artifact)
        observed = _required_sha256(path)

        if observed != expected:
            raise ValueError(
                "Membership source fingerprint mismatch: "
                f"{artifact} expected={expected} observed={observed}"
            )

        verified[artifact] = observed

    return verified


def _assert_lineage_isolation(
    members: list[dict[str, Any]],
) -> None:
    partitions: dict[str, set[str]] = {}

    for member in members:
        lineage = member.get("lineage_id")
        partition = member.get("partition")

        if not isinstance(lineage, str) or not lineage.strip():
            raise ValueError(
                "Membership member has no lineage_id."
            )

        if partition not in {"train", "validation"}:
            raise ValueError(
                "Membership member has invalid partition."
            )

        partitions.setdefault(lineage, set()).add(partition)

    leaking = sorted(
        lineage
        for lineage, values in partitions.items()
        if len(values) > 1
    )

    if leaking:
        raise ValueError(
            "Train/validation lineage leakage detected: "
            + ", ".join(leaking)
        )


def _curriculum_lesson_records(
    *,
    member: dict[str, Any],
    source: dict[str, Any],
    system_prompt: str,
    registry,
) -> tuple[HubSftRecord, HubDpoRecord | None]:
    chosen = source.get("chosen_response")
    rejected = source.get("rejected_response")
    request = source.get("user_request")

    if not (
        isinstance(chosen, str)
        and chosen.strip()
        and isinstance(request, str)
        and request.strip()
    ):
        raise ValueError(
            "Curriculum lesson is missing request/chosen response."
        )

    validate_hub_response_contract(
        response=chosen,
        registry=registry,
    )

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request},
    ]

    common = {
        "member_id": member["member_id"],
        "partition": member["partition"],
        "role": member["role"],
        "source_kind": "curriculum_lesson",
        "prompt_messages": prompt_messages,
        "chosen": chosen,
        "source_id": member["source_id"],
        "source_artifact": member["source_artifact"],
        "source_artifact_sha256":
            member["source_artifact_sha256"],
        "lineage_id": member["lineage_id"],
    }

    sft = HubSftRecord(
        record_id=(
            "sft-"
            + sha256_text(canonical_json(common))[:24]
        ),
        **common,
    )

    dpo = None

    if (
        isinstance(rejected, str)
        and rejected.strip()
        and rejected.strip() != chosen.strip()
    ):
        payload = {
            **common,
            "rejected": rejected.strip(),
        }

        dpo = HubDpoRecord(
            record_id=(
                "dpo-"
                + sha256_text(
                    canonical_json(payload)
                )[:24]
            ),
            **payload,
        )

    return sft, dpo


def _hub_preference_records(
    *,
    member: dict[str, Any],
    source: dict[str, Any],
    environment: dict[str, Any],
) -> tuple[HubSftRecord, HubDpoRecord]:
    prompt_messages = source.get("prompt_messages")
    chosen = source.get("chosen")
    rejected = source.get("rejected")

    if not isinstance(prompt_messages, list) or not prompt_messages:
        raise ValueError(
            "Hub preference has no prompt_messages."
        )

    if not all(
        isinstance(message, dict)
        and isinstance(message.get("role"), str)
        and isinstance(message.get("content"), str)
        for message in prompt_messages
    ):
        raise ValueError(
            "Hub preference prompt_messages are malformed."
        )

    if (
        not isinstance(chosen, str)
        or not chosen.strip()
        or not isinstance(rejected, str)
        or not rejected.strip()
    ):
        raise ValueError(
            "Hub preference is missing chosen/rejected response."
        )

    if source.get("training_authorized") is not False:
        raise ValueError(
            "Hub preference training_authorized flag must remain false."
        )

    if source.get("source_attempt_validation_status") != "accepted":
        raise ValueError(
            "Hub preference source attempt was not accepted."
        )

    current_system_sha = environment["system_prompt_sha256"]

    if source.get("source_system_prompt_sha256") != current_system_sha:
        raise ValueError(
            "Hub preference was captured under a different "
            "system prompt; refusing stale preference training."
        )

    if (
        source.get("source_capability_catalog_sha256")
        != environment["capability_catalog_sha256"]
    ):
        raise ValueError(
            "Hub preference was captured under a different "
            "capability catalog; refusing stale preference training."
        )

    first = prompt_messages[0]

    if (
        first.get("role") != "system"
        or sha256_text(first.get("content", "")) != current_system_sha
    ):
        raise ValueError(
            "Hub preference exact historical prompt does not "
            "match the current trusted Hub prompt."
        )

    validate_hub_response_contract(
        response=chosen,
        registry=environment["registry"],
    )

    common = {
        "member_id": member["member_id"],
        "partition": member["partition"],
        "role": member["role"],
        "source_kind": "hub_preference",
        "prompt_messages": prompt_messages,
        "chosen": chosen,
        "source_id": member["source_id"],
        "source_artifact": member["source_artifact"],
        "source_artifact_sha256":
            member["source_artifact_sha256"],
        "lineage_id": member["lineage_id"],
    }

    sft = HubSftRecord(
        record_id=(
            "sft-"
            + sha256_text(canonical_json(common))[:24]
        ),
        **common,
    )

    dpo_payload = {
        **common,
        "rejected": rejected,
    }

    dpo = HubDpoRecord(
        record_id=(
            "dpo-"
            + sha256_text(
                canonical_json(dpo_payload)
            )[:24]
        ),
        **dpo_payload,
    )

    return sft, dpo


def _write_partition(
    *,
    directory: Path,
    name: str,
    values: list[BaseModel],
) -> str:
    return immutable_write_jsonl(
        directory / f"{name}.jsonl",
        values,
    )


def materialize_hub_training(
    *,
    plan_directory: Path,
    target_model_key: str,
    base_model_path: Path,
    output_root: Path = DEFAULT_HUB_MATERIALIZATION_ROOT,
    agent_directory: Path | None = None,
) -> HubTrainingMaterializationResult:
    plan_directory = (
        plan_directory.expanduser().resolve()
    )

    manifest_path = (
        plan_directory / "manifest.json"
    )
    members_path = (
        plan_directory / "members.jsonl"
    )

    if not manifest_path.is_file() or not members_path.is_file():
        raise ValueError(
            "Training membership plan is incomplete."
        )

    plan_manifest = _read_json(manifest_path)
    members = _read_jsonl(members_path)

    if plan_manifest.get("target_component") != "hub":
        raise ValueError(
            "Phase-5 complete trainer currently targets the Hub. "
            "Specialist training remains on the existing specialist "
            "DPO/QLoRA pipeline."
        )

    if plan_manifest.get("readiness_ready") is not True:
        raise PermissionError(
            "Membership plan is not curriculum-ready for training."
        )

    if (
        plan_manifest.get("training_authorized") is not False
        or plan_manifest.get("promotion_authorized") is not False
    ):
        raise ValueError(
            "Membership plan governance flags are invalid."
        )

    observed_members_sha = _required_sha256(members_path)

    if (
        plan_manifest.get("membership_sha256")
        != observed_members_sha
    ):
        raise ValueError(
            "Membership records SHA-256 mismatch."
        )

    _assert_lineage_isolation(members)

    verified_fingerprints = _verify_source_fingerprints(
        plan_manifest
    )

    base_model_path = (
        base_model_path.expanduser().resolve()
    )

    if not base_model_path.is_dir():
        raise ValueError(
            "Base model directory does not exist: "
            f"{base_model_path}"
        )

    base_model_sha = fingerprint_directory(
        base_model_path
    )

    if agent_directory is None:
        agent_directory = (
            REPOSITORY_ROOT
            / "subagents"
            / "agents"
        )

    environment = build_hub_training_environment(
        agent_directory=agent_directory
    )

    current_chapter_id = str(
        plan_manifest.get("current_chapter_id")
    )

    curriculum_prompt = _curriculum_prompt_for_chapter(
        current_chapter_id,
        runtime_system_prompt=environment["system_prompt"],
    )

    curriculum_prompt_sha256 = sha256_text(
        curriculum_prompt
    )

    source_artifacts = {
        member.get("source_artifact")
        for member in members
        if isinstance(
            member.get("source_artifact"),
            str,
        )
    }

    source_index = _index_source_records(
        source_artifacts=source_artifacts
    )

    sft_train = []
    sft_validation = []
    dpo_train = []
    dpo_validation = []

    excluded: Counter[str] = Counter()
    contract_validated = 0

    for member in members:
        kind = member.get("member_kind")
        source_artifact = member.get("source_artifact")
        source_id = member.get("source_id")

        if (
            not isinstance(source_artifact, str)
            or not isinstance(source_id, str)
        ):
            raise ValueError(
                "Membership member has invalid source identity."
            )

        if (
            member.get("source_artifact_sha256")
            != verified_fingerprints.get(source_artifact)
        ):
            raise ValueError(
                "Membership member source SHA does not match "
                f"verified plan source: {member.get('member_id')}"
            )

        source = source_index.get(
            (source_artifact, source_id)
        )

        if source is None:
            raise ValueError(
                "Could not resolve membership source record: "
                f"{source_artifact}:{source_id}"
            )

        if kind == "curriculum_lesson":
            sft, dpo = _curriculum_lesson_records(
                member=member,
                source=source,
                system_prompt=curriculum_prompt,
                registry=environment["registry"],
            )
            contract_validated += 1

        elif kind == "preference":
            if source.get("target_component") != "hub":
                raise ValueError(
                    "Exact Hub membership references a non-Hub "
                    "preference record."
                )

            sft, dpo = _hub_preference_records(
                member=member,
                source=source,
                environment=environment,
            )
            contract_validated += 1

        else:
            raise ValueError(
                "Exact membership contains an unsupported member kind: "
                f"{kind!r}. Phase-5 materialization refuses to silently "
                "drop membership records."
            )

        partition = member.get("partition")

        if partition == "train":
            sft_train.append(sft)

            if dpo is not None:
                dpo_train.append(dpo)

        elif partition == "validation":
            sft_validation.append(sft)

            if dpo is not None:
                dpo_validation.append(dpo)

        else:
            raise ValueError(
                "Membership member has invalid partition."
            )

    if not sft_train:
        raise ValueError(
            "No SFT training records were materialized."
        )

    if not sft_validation:
        raise ValueError(
            "No SFT validation records were materialized."
        )

    if dpo_train and not dpo_validation:
        raise ValueError(
            "DPO training examples exist but no DPO validation "
            "examples were materialized."
        )

    identity_payload = {
        "source_plan_id":
            plan_manifest.get("plan_id"),
        "source_plan_manifest_sha256":
            _required_sha256(manifest_path),
        "source_plan_members_sha256":
            observed_members_sha,
        "target_model_key":
            target_model_key,
        "base_model_sha256":
            base_model_sha,
        "system_prompt_sha256":
            environment["system_prompt_sha256"],
        "capability_catalog_sha256":
            environment["capability_catalog_sha256"],
        "agent_definitions_sha256":
            environment["agent_definitions_sha256"],
        "curriculum_prompt_sha256":
            curriculum_prompt_sha256,
        "member_ids": [
            member.get("member_id")
            for member in members
        ],
    }

    materialization_id = (
        "phase5-materialization-"
        + sha256_text(
            canonical_json(identity_payload)
        )[:24]
    )

    target = (
        output_root.expanduser().resolve()
        / materialization_id
    )

    manifest_target = target / "manifest.json"

    if manifest_target.is_file():
        manifest = (
            HubTrainingMaterializationManifest
            .model_validate_json(
                manifest_target.read_text(
                    encoding="utf-8"
                )
            )
        )

        return HubTrainingMaterializationResult(
            manifest=manifest,
            output_directory=str(target),
        )

    target.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        sft_train_sha = _write_partition(
            directory=target,
            name="sft-train",
            values=sft_train,
        )
        sft_validation_sha = _write_partition(
            directory=target,
            name="sft-validation",
            values=sft_validation,
        )
        dpo_train_sha = _write_partition(
            directory=target,
            name="dpo-train",
            values=dpo_train,
        )
        dpo_validation_sha = _write_partition(
            directory=target,
            name="dpo-validation",
            values=dpo_validation,
        )

        ready = (
            bool(sft_train)
            and bool(sft_validation)
            and (
                not dpo_train
                or bool(dpo_validation)
            )
            and contract_validated
            == len(sft_train) + len(sft_validation)
        )

        manifest = HubTrainingMaterializationManifest(
            materialization_id=materialization_id,
            created_at=_utc_now(),
            target_component="hub",
            target_model_key=target_model_key,
            source_plan_id=str(
                plan_manifest.get("plan_id")
            ),
            source_plan_directory=str(plan_directory),
            source_plan_manifest_sha256=(
                _required_sha256(manifest_path)
            ),
            source_plan_members_sha256=(
                observed_members_sha
            ),
            curriculum_id=str(
                plan_manifest.get("curriculum_id")
            ),
            curriculum_version=str(
                plan_manifest.get("curriculum_version")
            ),
            chapter_id=str(
                plan_manifest.get("current_chapter_id")
            ),
            base_model_path=str(base_model_path),
            base_model_sha256=base_model_sha,
            hub_system_prompt_sha256=(
                environment["system_prompt_sha256"]
            ),
            hub_capability_catalog_sha256=(
                environment["capability_catalog_sha256"]
            ),
            hub_agent_definitions_sha256=(
                environment["agent_definitions_sha256"]
            ),
            curriculum_prompt_sha256=(
                curriculum_prompt_sha256
            ),
            source_fingerprint_count=len(
                verified_fingerprints
            ),
            source_fingerprints_verified=True,
            sft_train_count=len(sft_train),
            sft_validation_count=len(sft_validation),
            dpo_train_count=len(dpo_train),
            dpo_validation_count=len(dpo_validation),
            sft_train_sha256=sft_train_sha,
            sft_validation_sha256=sft_validation_sha,
            dpo_train_sha256=dpo_train_sha,
            dpo_validation_sha256=dpo_validation_sha,
            contract_validated_record_count=(
                contract_validated
            ),
            excluded_counts=dict(
                sorted(excluded.items())
            ),
            lineage_isolated=True,
            ready_for_training=ready,
            training_authorized=False,
            promotion_authorized=False,
        )

        immutable_write_json(
            manifest_target,
            manifest,
        )

    except Exception:
        shutil.rmtree(
            target,
            ignore_errors=True,
        )
        raise

    return HubTrainingMaterializationResult(
        manifest=manifest,
        output_directory=str(target),
    )


def find_latest_ready_hub_plan(
    *,
    membership_root: Path = DEFAULT_MEMBERSHIP_ROOT,
) -> Path:
    root = (
        membership_root.expanduser().resolve()
        / "hub"
    )

    if not root.is_dir():
        raise ValueError(
            "No Hub membership-plan directory exists."
        )

    candidates = []

    for manifest_path in root.glob(
        "*/manifest.json"
    ):
        try:
            manifest = _read_json(manifest_path)
        except Exception:
            continue

        if (
            manifest.get("target_component") == "hub"
            and manifest.get("readiness_ready") is True
        ):
            candidates.append(
                (
                    manifest.get("created_at", ""),
                    manifest_path.parent,
                )
            )

    if not candidates:
        raise ValueError(
            "No curriculum-ready Hub membership plan exists."
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            str(item[1]),
        )
    )

    return candidates[-1][1]
