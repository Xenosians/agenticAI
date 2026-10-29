"""Owned resource selection preflight; candidate admission remains separate.

The extension registry holds inference connections and reviewed dataset splits.
A check is `ready` only when a candidate SFT run can start here: the model is
owned local weights the host can read, the method is SFT with LoRA or QLoRA,
and the dataset version is current. DPO and PPO keep their own open gates.
A ready check is still not a training job, and never changes serving.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Callable

from pydantic import Field

from .contracts import Contract, Digest, Method, Role, Tuning

ResourceId = Annotated[str, Field(strict=True, pattern=r"^[a-f0-9]{32}$")]
# Installed profiles and admitted uploads are owned weights; the rest are outbound connections.
LOCAL_BACKENDS = {"local", "uploaded"}
BACKENDS = LOCAL_BACKENDS | {"http", "pi", "opencode"}
# Candidate runs this host can start from the web: SFT with adapter tuning only.
RUNNABLE_TUNING = {"lora", "qlora"}
LATER_GATES = ["fixed_evaluation_required", "separate_promotion_required"]


def model_weights(row: dict) -> Path | None:
    """The local weights directory of an owned model, or None when the host has none."""
    if row.get("backend") == "uploaded":
        path = row.get("model_path")
        if isinstance(path, str) and path and (Path(path) / ".sealed").is_file():
            return Path(path)
        return None
    if row.get("backend") == "local":
        from config.settings import get_settings
        profile = get_settings().model_profiles.get(str(row.get("model", "")))
        if profile is None or not profile.enabled or profile.adapter_path is not None or profile.model_path is None:
            return None
        path = Path(profile.model_path).expanduser()
        return path if (path / "config.json").is_file() else None
    return None


class SelectionRequest(Contract):
    method: Method
    role: Role
    tuning: Tuning
    model_id: ResourceId
    dataset_id: ResourceId
    dataset_sha256: Digest


class LearningSelectionError(ValueError):
    """Only these curated reason codes may cross the API boundary."""


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _owner(owner: str) -> None:
    if not isinstance(owner, str) or not owner.strip() or len(owner) > 200:
        raise LearningSelectionError("invalid_learning_request")


def _model(row: dict, weights: Callable[[dict], Path | None] | None = None) -> dict:
    if (row.get("kind") != "model" or row.get("backend") not in BACKENDS or
            not isinstance(row.get("id"), str) or not re.fullmatch(r"[a-f0-9]{32}", row["id"]) or
            not isinstance(row.get("name"), str) or not 0 < len(row["name"]) <= 120 or
            row.get("state") not in {"ready", "pending", "rejected"}):
        raise LearningSelectionError("learning_resource_unavailable")
    source = "local" if row["backend"] in LOCAL_BACKENDS else "outbound"
    available = source == "local" and row["state"] == "ready" and weights is not None and weights(row) is not None
    return {"id": row["id"], "name": row["name"], "backend": row["backend"], "state": row["state"],
            "source_kind": source, "training_available": available}


def _dataset(row: dict) -> dict:
    if row.get("kind") != "dataset" or row.get("purpose") != "training":
        raise LearningSelectionError("learning_dataset_not_training")
    if (row.get("state") != "ready" or not isinstance(row.get("sha256"), str) or
            not re.fullmatch(r"[a-f0-9]{64}", row["sha256"]) or
            not isinstance(row.get("name"), str) or not 0 < len(row["name"]) <= 120 or
            type(row.get("version")) is not int or row["version"] < 1 or
            not all(isinstance(row.get(key), str) and re.fullmatch(r"[a-f0-9]{32}", row[key])
                    for key in ("id", "lineage_id")) or
            not isinstance(row.get("license"), str) or not row["license"].strip() or
            not isinstance(row.get("source"), str) or not row["source"].strip()):
        raise LearningSelectionError("learning_resource_unavailable")
    content = row.get("content")
    if type(content) is not dict or set(content) != {"train", "validation"}:
        raise LearningSelectionError("learning_training_membership_invalid")
    seen = set()
    for records in content.values():
        if type(records) is not list or not records:
            raise LearningSelectionError("learning_training_membership_invalid")
        for record in records:
            if type(record) is not dict or set(record) != {"messages"}:
                raise LearningSelectionError("learning_training_membership_invalid")
            messages = record["messages"]
            if (type(messages) is not list or len(messages) < 2 or
                    any(type(message) is not dict or set(message) != {"role", "content"} or
                        message["role"] not in {"system", "user", "assistant"} or
                        not isinstance(message["content"], str) or not message["content"].strip()
                        for message in messages) or messages[-1]["role"] != "assistant"):
                raise LearningSelectionError("learning_training_membership_invalid")
            identity = _digest(messages[:-1])
            if identity in seen:
                raise LearningSelectionError("learning_training_membership_invalid")
            seen.add(identity)
    if len(seen) < 10:
        raise LearningSelectionError("learning_training_membership_invalid")
    newer = row.get("superseded_by")
    if newer is not None and (not isinstance(newer, str) or not re.fullmatch(r"[a-f0-9]{32}", newer)):
        raise LearningSelectionError("learning_resource_unavailable")
    return {"id": row["id"], "name": row["name"], "purpose": "training", "state": row["state"],
            "sha256": row["sha256"], "lineage_id": row["lineage_id"], "version": row["version"],
            "train_count": len(content["train"]), "validation_count": len(content["validation"]),
            "superseded_by": newer}


def learning_catalog(owner: str, store, *, weights: Callable[[dict], Path | None] = model_weights) -> dict:
    _owner(owner)
    from learning.training.methods import METHODS
    models, datasets = [], []
    for row in store.list(owner):
        if row.get("kind") == "model":
            models.append(_model(row, weights))
        elif row.get("kind") == "dataset" and row.get("purpose") == "training":
            datasets.append(_dataset(row))
    if len({row["id"] for row in models + datasets}) != len(models + datasets):
        raise LearningSelectionError("learning_resource_unavailable")
    return {"schema": "learning-catalog.v2",
            "methods": [{"id": key, **METHODS[key]} for key in ("sft", "dpo", "ppo")],
            "models": models, "datasets": datasets, "promotion_enabled": False}


def run_blockers(request: SelectionRequest, model: dict, has_weights: bool) -> list[str]:
    """What still prevents a candidate run of this selection on this host (empty when it can start)."""
    blockers = []
    if model["source_kind"] == "outbound":
        blockers.append("outbound_connection_is_inference_only")
    elif not has_weights:
        blockers.append("local_weights_unavailable")
    if request.method == "dpo":
        blockers += ["reviewed_preference_partitions_required", "host_dispatch_not_integrated"]
    elif request.method == "ppo":
        blockers += ["evaluated_adapter_checkpoint_required", "independent_simulator_episodes_required",
                     "host_dispatch_not_integrated"]
    elif request.tuning not in RUNNABLE_TUNING:
        blockers.append("full_tuning_not_offered")
    return sorted(blockers)


def selection_preflight(owner: str, request: SelectionRequest, store, *,
                        weights: Callable[[dict], Path | None] = model_weights) -> dict:
    _owner(owner)
    request = SelectionRequest.model_validate(request.model_dump())
    from learning.training.methods import validate_method_role
    try:
        method = validate_method_role(request.method, request.role, request.tuning)
    except ValueError:
        raise LearningSelectionError("invalid_learning_request") from None
    # The owner enters only through the authenticated host. Missing and foreign
    # identities have the same result and never disclose another owner's data.
    try:
        rows = {row["id"]: row for row in store.list(owner)}
        model_row, dataset_row = rows[request.model_id], rows[request.dataset_id]
    except (KeyError, ValueError, TypeError):
        raise LearningSelectionError("learning_resource_unavailable") from None
    model, dataset = _model(model_row), _dataset(dataset_row)
    if model["state"] != "ready":
        raise LearningSelectionError("learning_resource_unavailable")
    if dataset["sha256"] != request.dataset_sha256:
        raise LearningSelectionError("learning_dataset_changed")
    if dataset["superseded_by"] is not None:
        raise LearningSelectionError("learning_dataset_superseded")
    has_weights = model["source_kind"] == "local" and weights(model_row) is not None
    blockers = run_blockers(request, model, has_weights)
    status = "blocked" if blockers else "ready"
    pinned_model = {key: model[key] for key in ("id", "backend", "source_kind", "state")}
    pinned_dataset = {key: dataset[key] for key in ("id", "sha256", "lineage_id", "version", "train_count", "validation_count")}
    # Binds current memberships as well as immutable-version metadata; a run must
    # present this exact digest, so a changed selection or host needs a new check.
    preflight_sha256 = _digest({"owner": owner, "request": request.model_dump(mode="json"),
                               "model": pinned_model, "dataset": pinned_dataset,
                               "membership": dataset_row["content"], "status": status, "blockers": blockers})
    return {"schema": "learning-selection-preflight.v2", "status": status, "preflight_sha256": preflight_sha256,
            "method": request.method, "role": request.role, "tuning": request.tuning,
            "model": pinned_model, "dataset": pinned_dataset, "entrypoint": method["entrypoint"],
            "blockers": blockers, "later_gates": list(LATER_GATES),
            "model_loaded": False, "live_tool_execution": False, "promotion_enabled": False}
