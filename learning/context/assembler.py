from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timezone

from learning.context.storage import canonical_json
from learning.context.models import LearningContextBundle, StructuredContextRecord


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _matches_lineage(
    record: StructuredContextRecord,
    *,
    trajectory_id: str | None,
    task_id: str | None,
    job_id: str | None,
) -> bool:
    if record.scope == "global":
        return record.kind == "human"
    if task_id and record.task_id == task_id:
        return True
    if trajectory_id and record.trajectory_id == trajectory_id:
        return True
    if job_id and record.job_id == job_id:
        return True
    return False


def assemble_context_bundle(
    records: list[StructuredContextRecord],
    *,
    trajectory_id: str | None = None,
    task_id: str | None = None,
    job_id: str | None = None,
    max_records: int = 96,
    max_payload_chars: int = 40_000,
    per_kind_limit: int = 24,
    include_all_records: bool = False,
) -> LearningContextBundle:
    """
    Assemble bounded contextual evidence.

    Normal callers should provide lineage identifiers. In that mode only
    matching task / trajectory / job records plus global human context are
    included.

    ``include_all_records`` exists only for compatibility helpers whose
    caller has already explicitly selected the records to bundle. Keeping
    it opt-in preserves lineage isolation for the primary API.
    """

    if max_records < 1 or max_payload_chars < 1 or per_kind_limit < 1:
        raise ValueError("context assembly limits must be positive")

    if include_all_records:
        eligible = list(records)
    else:
        eligible = [
            item
            for item in records
            if _matches_lineage(
                item,
                trajectory_id=trajectory_id,
                task_id=task_id,
                job_id=job_id,
            )
        ]

    selected: list[StructuredContextRecord] = []
    kind_counts: Counter[str] = Counter()
    used_chars = 0

    for item in reversed(eligible):
        if len(selected) >= max_records:
            break
        if kind_counts[item.kind] >= per_kind_limit:
            continue
        payload_blob = canonical_json(item.payload)
        if selected and used_chars + len(payload_blob) > max_payload_chars:
            continue
        selected.append(item)
        kind_counts[item.kind] += 1
        used_chars += len(payload_blob)

    selected.reverse()
    serialized = canonical_json(
        [item.model_dump(mode="json", by_alias=True) for item in selected]
    )

    return LearningContextBundle(
        created_at=_utc_now(),
        trajectory_id=trajectory_id,
        task_id=task_id,
        job_id=job_id,
        record_count=len(selected),
        kind_counts=dict(kind_counts),
        content_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        records=selected,
    )
