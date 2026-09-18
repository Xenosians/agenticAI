from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from learning.context.recorder import load_context_records
from learning.continual.materialization import build_materialization_plan
from learning.continual.replay import ReplayPolicy, select_replay
from learning.continual.storage import canonical_json, immutable_write_json, immutable_write_jsonl, sha256_file
from learning.continual.triggers import TrainingTriggerPolicy, evaluate_training_trigger
from learning.continual.types import ContinualCycleManifest
from learning.continual.weak_supervision import classify_trajectories
from learning.continual.windows import build_evidence_window
from learning.curriculum.scheduler import build_curriculum_plan
from learning.curation.corpus_analysis import load_corrections, load_trajectories
from learning.curation.reviews import load_reviews
from learning.paths import CORRECTIONS_PATH, REVIEWS_PATH, RUNTIME_LEARNING_ROOT, TRAJECTORIES_PATH


DEFAULT_CONTINUAL_ROOT = RUNTIME_LEARNING_ROOT / "continual"
DEFAULT_CONTEXT_RECORDS = DEFAULT_CONTINUAL_ROOT / "context-records.jsonl"
# Backward compatibility with the first generated package.
DEFAULT_CONTEXT_EVENTS = DEFAULT_CONTEXT_RECORDS
PHASE5_ALGORITHM_VERSION = "phase5-contextual-curriculum-v2"


class ContinualLearningPolicy(BaseModel):
    min_gold_records: int = Field(default=16, ge=1)
    min_recent_gold_records: int = Field(default=4, ge=0)
    min_gold_domains: int = Field(default=2, ge=1)
    min_novelty_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    min_failure_clusters: int = Field(default=1, ge=0)
    max_replay_records: int = Field(default=256, ge=1)
    recent_window_size: int = Field(default=64, ge=1)
    historical_window_size: int = Field(default=192, ge=0)
    historical_fraction: float = Field(default=0.35, ge=0.0, le=0.9)
    max_domain_fraction: float = Field(default=0.60, gt=0.0, le=1.0)
    replay_seed: str = "continual-replay-v2"
    curriculum_seed: str = "continual-curriculum-v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ContinualLearningCycle:
    """
    Governed Phase-5 evidence controller.

    Responsibilities:
        contextualize immutable runtime evidence
        weak-label for review/replay
        build recent + historical windows
        classify domain and difficulty
        construct a balanced curriculum
        evaluate batch-readiness triggers
        plan DPO/SFT candidate materialization
        emit immutable artifacts

    Non-responsibilities:
        no trainer.train()
        no live weight mutation
        no automatic adapter activation
        no bypass around SemanticGuard or ToolGateway
    """

    def __init__(
        self,
        *,
        trajectories_path: Path = TRAJECTORIES_PATH,
        corrections_path: Path = CORRECTIONS_PATH,
        reviews_path: Path = REVIEWS_PATH,
        context_path: Path = DEFAULT_CONTEXT_RECORDS,
        output_root: Path = DEFAULT_CONTINUAL_ROOT / "cycles",
        policy: ContinualLearningPolicy | None = None,
    ) -> None:
        self.trajectories_path = trajectories_path.expanduser().resolve()
        self.corrections_path = corrections_path.expanduser().resolve()
        self.reviews_path = reviews_path.expanduser().resolve()
        self.context_path = context_path.expanduser().resolve()
        self.output_root = output_root.expanduser().resolve()
        self.policy = policy or ContinualLearningPolicy()

    def _cycle_id(self) -> str:
        payload = {
            "trajectories": sha256_file(self.trajectories_path),
            "corrections": sha256_file(self.corrections_path),
            "reviews": sha256_file(self.reviews_path),
            "context": sha256_file(self.context_path),
            "policy": self.policy.model_dump(mode="json"),
            "algorithm_version": PHASE5_ALGORITHM_VERSION,
        }
        digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        return f"cycle-{digest[:20]}"

    def run(self) -> ContinualCycleManifest:
        trajectories = load_trajectories(self.trajectories_path)
        corrections = load_corrections(self.corrections_path)
        reviews = load_reviews(self.reviews_path)
        context_records = load_context_records(self.context_path)

        signals = classify_trajectories(
            trajectories=trajectories,
            corrections=corrections,
            reviews=reviews,
        )

        window = build_evidence_window(
            trajectories,
            recent_size=self.policy.recent_window_size,
            historical_size=self.policy.historical_window_size,
        )

        replay = select_replay(
            trajectories=trajectories,
            signals=signals,
            policy=ReplayPolicy(
                max_records=self.policy.max_replay_records,
                historical_fraction=self.policy.historical_fraction,
                seed=self.policy.replay_seed,
                max_domain_fraction=self.policy.max_domain_fraction,
            ),
        )

        curriculum = build_curriculum_plan(
            trajectories=trajectories,
            signals=signals,
            corrections=corrections,
            context_records=context_records,
            max_records=self.policy.max_replay_records,
            max_domain_fraction=self.policy.max_domain_fraction,
            seed=self.policy.curriculum_seed,
        )

        trigger = evaluate_training_trigger(
            signals=signals,
            curriculum_plan=curriculum,
            recent_trajectory_ids=set(window.recent_trajectory_ids),
            policy=TrainingTriggerPolicy(
                min_gold_records=self.policy.min_gold_records,
                min_recent_gold_records=self.policy.min_recent_gold_records,
                min_gold_domains=self.policy.min_gold_domains,
                min_novelty_fraction=self.policy.min_novelty_fraction,
                min_failure_clusters=self.policy.min_failure_clusters,
                max_domain_fraction=self.policy.max_domain_fraction,
            ),
        )

        materialization = build_materialization_plan(
            signals=signals,
            corrections=corrections,
        )

        signal_index = {item.trajectory_id: item for item in signals}
        candidate_index = {item.trajectory_id: item for item in curriculum.candidates}
        tier_counts = Counter(item.tier for item in signals)

        review_queue = []
        for signal in signals:
            if signal.tier not in {"silver", "bronze"}:
                continue
            candidate = candidate_index.get(signal.trajectory_id)
            review_queue.append(
                {
                    "trajectory_id": signal.trajectory_id,
                    "tier": signal.tier,
                    "reward_score": signal.reward_score,
                    "reasons": signal.reasons,
                    "domain": candidate.primary_domain if candidate else "unknown",
                    "difficulty": candidate.difficulty if candidate else "unknown",
                    "failure_clusters": candidate.failure_clusters if candidate else [],
                }
            )

        contextualized_ids = {
            item.trajectory_id
            for item in context_records
            if item.trajectory_id is not None
        }

        if trigger.ready:
            recommended_action = (
                "materialize only provenance-verified gold candidates through the existing "
                "dataset/export and Phase-4 DPO/QLoRA gates; evaluate before promotion"
            )
        elif tier_counts.get("silver", 0) > 0:
            recommended_action = "review silver evidence and collect missing trusted corrections; do not auto-train"
        else:
            recommended_action = "collect more diverse real runtime failures/context before training"

        cycle_id = self._cycle_id()
        directory = self.output_root / cycle_id
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            return ContinualCycleManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))

        directory.mkdir(parents=True, exist_ok=False)
        immutable_write_jsonl(directory / "signals.jsonl", signals)
        immutable_write_jsonl(
            directory / "replay.jsonl",
            [
                {
                    "trajectory_id": item_id,
                    "tier": signal_index[item_id].tier,
                    "domain": candidate_index[item_id].primary_domain if item_id in candidate_index else "unknown",
                    "difficulty": candidate_index[item_id].difficulty if item_id in candidate_index else "unknown",
                }
                for item_id in replay.selected_trajectory_ids
                if item_id in signal_index
            ],
        )
        immutable_write_jsonl(directory / "review-queue.jsonl", review_queue)
        immutable_write_json(directory / "curriculum.json", curriculum)
        immutable_write_json(directory / "trigger.json", trigger)
        immutable_write_json(directory / "materialization-plan.json", materialization)
        immutable_write_json(directory / "evidence-window.json", window)

        manifest = ContinualCycleManifest(
            cycle_id=cycle_id,
            created_at=_utc_now(),
            output_directory=str(directory),
            trajectory_sha256=sha256_file(self.trajectories_path),
            corrections_sha256=sha256_file(self.corrections_path),
            reviews_sha256=sha256_file(self.reviews_path),
            context_sha256=sha256_file(self.context_path),
            trajectory_count=len(trajectories),
            correction_count=len(corrections),
            review_count=len(reviews),
            context_event_count=len(context_records),
            context_record_count=len(context_records),
            contextualized_trajectory_count=len(contextualized_ids),
            tier_counts=dict(tier_counts),
            selected_replay_count=len(replay.selected_trajectory_ids),
            selected_domain_counts=curriculum.selection.domain_counts,
            curriculum_difficulty_counts=curriculum.selection.difficulty_counts,
            failure_cluster_counts=curriculum.selection.failure_cluster_counts,
            recent_window_count=window.recent_count,
            historical_window_count=window.historical_count,
            min_gold_records=self.policy.min_gold_records,
            min_gold_domains=self.policy.min_gold_domains,
            train_ready=trigger.ready,
            trigger_checks=trigger.checks,
            train_block_reasons=trigger.reasons,
            review_queue_count=len(review_queue),
            dpo_candidate_count=len(materialization.dpo_candidate_trajectory_ids),
            sft_candidate_count=len(materialization.sft_candidate_trajectory_ids),
            recommended_action=recommended_action,
            training_executed=False,
            auto_promotion_performed=False,
        )
        immutable_write_json(manifest_path, manifest)
        return manifest
