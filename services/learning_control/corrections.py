"""Accepted corrections as the owner's training data (FR-085/089).

A correction an owner wrote on one of their task answers, once a reviewer
accepted it, pairs that request with the answer it should have had. Saving the
owner's accepted corrections adds the next version of one training dataset
through the ordinary training admission (privacy scan, fixed training and
validation splits, version identity). Saving never starts or changes a run.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Annotated

from pydantic import Field, field_validator

from .application import LearningSelectionError
from .contracts import Contract

SCHEMA = "learning-corrections.v1"
DATASET_NAME = "Accepted corrections"
DATASET_SOURCE = "Task corrections accepted in review"
DATASET_LICENSE = "Written by the owner and accepted by a reviewer"
MAX_CORRECTIONS = 500


class AcceptedCorrection(Contract):
    feedback_id: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{32}$")]
    request: str = Field(min_length=1, max_length=20_000)
    correction: str = Field(min_length=1, max_length=2_000)
    reviewed_at: Annotated[str, Field(strict=True,
                                      pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")]

    @field_validator("reviewed_at")
    @classmethod
    def _real_time(cls, value: str) -> str:
        reviewed(value)
        return value


def reviewed(value: str) -> datetime:
    """When a reviewer accepted the correction (UTC); an impossible time is refused."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _training_rows(corrections: list[AcceptedCorrection]) -> tuple[list[tuple[str, str]], int]:
    """(feedback id, JSONL row) for the latest accepted correction of each request, in
    request order, and how many corrections were withheld by the privacy scan."""
    from services.extensions.store import sensitive_content
    latest: dict[str, tuple[tuple[datetime, str], str]] = {}
    withheld = 0
    for item in corrections:
        request, answer = item.request.strip(), item.correction.strip()
        line = json.dumps({"messages": [{"role": "user", "content": request},
                                        {"role": "assistant", "content": answer}]}, ensure_ascii=False)
        if not request or not answer or sensitive_content(line) is not None:
            withheld += 1
            continue
        order = (reviewed(item.reviewed_at), item.feedback_id)
        if request not in latest or order > latest[request][0]:
            latest[request] = (order, line)
    return [(latest[key][0][1], latest[key][1]) for key in sorted(latest)], withheld


def _version(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {"id": row["id"], "name": row["name"], "version": row["version"], "lineage_id": row["lineage_id"],
            "sha256": row["sha256"], "train_count": len(row["content"]["train"]),
            "validation_count": len(row["content"]["validation"])}


def accepted_corrections(owner: str, corrections: list[AcceptedCorrection], store, *, apply: bool) -> dict:
    """Preview, or with ``apply`` save, the owner's accepted corrections as training data.

    ``status`` is ``ready`` (can be saved), ``too_few`` (fewer distinct requests
    than training admission needs), ``unchanged`` (the latest saved version
    already holds exactly these rows) or ``saved``. ``dataset`` is the latest
    saved version, or the version just saved.
    """
    from services.extensions.store import TRAINING_MINIMUM_INPUTS
    if len(corrections) > MAX_CORRECTIONS:
        raise LearningSelectionError("invalid_learning_request")
    rows, withheld = _training_rows(corrections)
    text = "\n".join(line for _, line in rows)
    previous = store.corrections_dataset(owner)
    if len(rows) < TRAINING_MINIMUM_INPUTS:
        status = "too_few"
    elif previous is not None and previous["sha256"] == hashlib.sha256(text.encode()).hexdigest():
        status = "unchanged"
    else:
        status = "ready"
    result = {"schema": SCHEMA, "status": status, "usable": len(rows), "withheld": withheld,
              "minimum": TRAINING_MINIMUM_INPUTS, "dataset": _version(previous)}
    if not apply:
        return result
    if status != "ready":
        raise LearningSelectionError(f"learning_corrections_{status}")
    try:
        saved = store.dataset(owner, name=DATASET_NAME, text=text, source=DATASET_SOURCE, license=DATASET_LICENSE,
                              purpose="training", reviewed=True, replaces=previous["id"] if previous else None,
                              corrections={"schema": SCHEMA, "feedback_ids": sorted(identity for identity, _ in rows),
                                           "withheld": withheld})
    except ValueError:
        # Admission refused the rows, or another save replaced the latest version first.
        raise LearningSelectionError("learning_corrections_refused") from None
    return {**result, "status": "saved", "dataset": _version(saved)}
