from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from learning.continual.storage import (
    append_jsonl,
    atomic_write_json,
    fingerprint_directory,
    immutable_write_json,
    read_json_model,
)
from learning.continual.types import ActiveCheckpointPointer, AdapterCheckpointManifest
from learning.curation.promotion_gate import PromotionGateStore
from learning.paths import PROMOTIONS_ROOT, RUNTIME_LEARNING_ROOT


DEFAULT_CHECKPOINT_ROOT = RUNTIME_LEARNING_ROOT / "continual" / "checkpoints"
DEFAULT_ACTIVE_POINTER = RUNTIME_LEARNING_ROOT / "continual" / "active-checkpoint.json"
DEFAULT_CHECKPOINT_HISTORY = RUNTIME_LEARNING_ROOT / "continual" / "checkpoint-history.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AdapterCheckpointStore:
    """Hash-pinned adapter registry plus promotion-gated active pointer."""

    def __init__(
        self,
        *,
        root: Path = DEFAULT_CHECKPOINT_ROOT,
        active_pointer: Path = DEFAULT_ACTIVE_POINTER,
        history_path: Path = DEFAULT_CHECKPOINT_HISTORY,
        promotion_root: Path = PROMOTIONS_ROOT,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.active_pointer = active_pointer.expanduser().resolve()
        self.history_path = history_path.expanduser().resolve()
        self.promotion_store = PromotionGateStore(root=promotion_root)

    def register(
        self,
        *,
        adapter_directory: Path,
        base_model_sha256: str,
        source_cycle_id: str,
        source_split_id: str,
        target_agent: str,
        target_model_key: str,
        label: str,
    ) -> AdapterCheckpointManifest:
        adapter_directory = adapter_directory.expanduser().resolve()
        adapter_sha256 = fingerprint_directory(adapter_directory)
        checkpoint_id = f"checkpoint-{uuid.uuid4().hex}"
        manifest = AdapterCheckpointManifest(
            checkpoint_id=checkpoint_id,
            created_at=_utc_now(),
            label=label.strip(),
            adapter_directory=str(adapter_directory),
            adapter_sha256=adapter_sha256,
            base_model_sha256=base_model_sha256.strip(),
            source_cycle_id=source_cycle_id.strip(),
            source_split_id=source_split_id.strip(),
            target_agent=target_agent.strip(),
            target_model_key=target_model_key.strip(),
        )
        immutable_write_json(self.root / checkpoint_id / "manifest.json", manifest)
        return manifest

    def load(self, checkpoint_id: str) -> AdapterCheckpointManifest:
        return read_json_model(self.root / checkpoint_id / "manifest.json", AdapterCheckpointManifest)

    def verify_adapter(self, checkpoint_id: str) -> AdapterCheckpointManifest:
        manifest = self.load(checkpoint_id)
        observed = fingerprint_directory(Path(manifest.adapter_directory))
        if observed != manifest.adapter_sha256:
            raise ValueError("Adapter SHA-256 verification failed")
        return manifest

    def active(self) -> ActiveCheckpointPointer | None:
        if not self.active_pointer.exists():
            return None
        return read_json_model(self.active_pointer, ActiveCheckpointPointer)

    def promote(self, *, checkpoint_id: str, suite: str, decision_id: str) -> ActiveCheckpointPointer:
        manifest = self.verify_adapter(checkpoint_id)
        promotion = self.promotion_store.load(suite=suite, decision_id=decision_id)
        if not promotion.decision.promotion_eligible:
            raise PermissionError("Promotion gate decision is not eligible")

        previous = self.active()
        pointer = ActiveCheckpointPointer(
            updated_at=_utc_now(),
            checkpoint_id=checkpoint_id,
            previous_checkpoint_id=previous.checkpoint_id if previous else None,
            promotion_decision_id=decision_id,
            rollback=False,
        )
        atomic_write_json(self.active_pointer, pointer)
        append_jsonl(self.history_path, pointer)

        updated = manifest.model_copy(
            update={
                "promotion_suite": suite,
                "promotion_decision_id": decision_id,
                "promotion_decision_sha256": promotion.decision_sha256,
                "promotion_eligible": True,
            }
        )
        # Keep the original registration immutable. Promotion metadata is separate.
        promotion_path = self.root / checkpoint_id / "promotion.json"
        immutable_write_json(promotion_path, updated)
        return pointer

    def rollback(self) -> ActiveCheckpointPointer:
        current = self.active()
        if current is None or current.previous_checkpoint_id is None:
            raise ValueError("No previous checkpoint is available for rollback")

        previous_manifest = self.verify_adapter(current.previous_checkpoint_id)
        previous_promotion_path = self.root / previous_manifest.checkpoint_id / "promotion.json"
        if not previous_promotion_path.is_file():
            raise PermissionError("Previous checkpoint was never promotion-gated")
        previous_promotion = read_json_model(previous_promotion_path, AdapterCheckpointManifest)
        if not previous_promotion.promotion_eligible or previous_promotion.promotion_decision_id is None:
            raise PermissionError("Previous checkpoint is not promotion-eligible")

        pointer = ActiveCheckpointPointer(
            updated_at=_utc_now(),
            checkpoint_id=previous_manifest.checkpoint_id,
            previous_checkpoint_id=current.checkpoint_id,
            promotion_decision_id=previous_promotion.promotion_decision_id,
            rollback=True,
        )
        atomic_write_json(self.active_pointer, pointer)
        append_jsonl(self.history_path, pointer)
        return pointer
