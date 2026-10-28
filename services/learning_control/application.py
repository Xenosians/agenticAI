"""Owned resource selection preflight; candidate admission remains separate.

The extension registry holds inference connections and reviewed dataset splits,
not the model/tokenizer/prompt/catalog/evaluation identities required by the
candidate planner. Do not invent those identities or report a training job.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated

from pydantic import Field

from .contracts import Contract, Digest, Method, Role, Tuning

ResourceId = Annotated[str, Field(strict=True, pattern=r"^[a-f0-9]{32}$")]
# Installed profiles and admitted uploads are owned weights; the rest are outbound connections.
LOCAL_BACKENDS = {"local", "uploaded"}
BACKENDS = LOCAL_BACKENDS | {"http", "pi", "opencode"}


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


def _model(row: dict) -> dict:
    if (row.get("kind") != "model" or row.get("backend") not in BACKENDS or
            not isinstance(row.get("id"), str) or not re.fullmatch(r"[a-f0-9]{32}", row["id"]) or
            not isinstance(row.get("name"), str) or not 0 < len(row["name"]) <= 120 or
            row.get("state") not in {"ready", "pending", "rejected"}):
        raise LearningSelectionError("learning_resource_unavailable")
    return {"id": row["id"], "name": row["name"], "backend": row["backend"], "state": row["state"],
            "source_kind": "local" if row["backend"] in LOCAL_BACKENDS else "outbound", "training_available": False}


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


def learning_catalog(owner: str, store) -> dict:
    _owner(owner)
    from learning.training.methods import METHODS
    models, datasets = [], []
    for row in store.list(owner):
        if row.get("kind") == "model":
            models.append(_model(row))
        elif row.get("kind") == "dataset" and row.get("purpose") == "training":
            datasets.append(_dataset(row))
    if len({row["id"] for row in models + datasets}) != len(models + datasets):
        raise LearningSelectionError("learning_resource_unavailable")
    return {"schema": "learning-catalog.v1",
            "methods": [{"id": key, **METHODS[key]} for key in ("sft", "dpo", "ppo")],
            "models": models, "datasets": datasets, "execution_enabled": False, "promotion_enabled": False}


def selection_preflight(owner: str, request: SelectionRequest, store) -> dict:
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
    blockers = ["trusted_model_identity_required", "trusted_training_configuration_required",
                "fixed_evaluation_required", "resource_unknown", "host_dispatch_not_integrated",
                "separate_promotion_required"]
    if model["source_kind"] == "outbound":
        blockers.append("outbound_connection_is_inference_only")
    if request.method == "dpo":
        blockers.append("reviewed_preference_partitions_required")
    elif request.method == "ppo":
        blockers.extend(["evaluated_adapter_checkpoint_required", "independent_simulator_episodes_required"])
    pinned_model = {key: model[key] for key in ("id", "backend", "source_kind", "state")}
    pinned_dataset = {key: dataset[key] for key in ("id", "sha256", "lineage_id", "version", "train_count", "validation_count")}
    # Include current memberships as well as immutable-version metadata. The
    # digest is an inspection identity, not a trainer admission certificate.
    preflight_sha256 = _digest({"owner": owner, "request": request.model_dump(mode="json"),
                               "model": pinned_model, "dataset": pinned_dataset,
                               "membership": dataset_row["content"], "blockers": sorted(blockers)})
    return {"schema": "learning-selection-preflight.v1", "status": "blocked", "preflight_sha256": preflight_sha256,
            "method": request.method, "role": request.role, "tuning": request.tuning,
            "model": pinned_model, "dataset": pinned_dataset, "entrypoint": method["entrypoint"],
            "resource_state": "unknown", "blockers": sorted(blockers), "execution_enabled": False,
            "model_loaded": False, "live_tool_execution": False, "promotion_enabled": False}
