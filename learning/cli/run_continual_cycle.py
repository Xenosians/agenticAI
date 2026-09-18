from __future__ import annotations

import argparse
import json
from pathlib import Path

from learning.continual.cycle import (
    DEFAULT_CONTEXT_RECORDS,
    DEFAULT_CONTINUAL_ROOT,
    ContinualLearningCycle,
    ContinualLearningPolicy,
)
from learning.paths import CORRECTIONS_PATH, REVIEWS_PATH, TRAJECTORIES_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one contextual Phase-5 continual-learning evidence cycle. "
            "This command does NOT train, update live weights, or auto-promote."
        )
    )
    parser.add_argument("--trajectories", type=Path, default=TRAJECTORIES_PATH)
    parser.add_argument("--corrections", type=Path, default=CORRECTIONS_PATH)
    parser.add_argument("--reviews", type=Path, default=REVIEWS_PATH)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT_RECORDS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_CONTINUAL_ROOT / "cycles")
    parser.add_argument("--min-gold-records", type=int, default=16)
    parser.add_argument("--min-recent-gold-records", type=int, default=4)
    parser.add_argument("--min-gold-domains", type=int, default=2)
    parser.add_argument("--min-novelty-fraction", type=float, default=0.25)
    parser.add_argument("--min-failure-clusters", type=int, default=1)
    parser.add_argument("--max-replay-records", type=int, default=256)
    parser.add_argument("--recent-window-size", type=int, default=64)
    parser.add_argument("--historical-window-size", type=int, default=192)
    parser.add_argument("--historical-fraction", type=float, default=0.35)
    parser.add_argument("--max-domain-fraction", type=float, default=0.60)
    parser.add_argument("--replay-seed", default="continual-replay-v2")
    parser.add_argument("--curriculum-seed", default="continual-curriculum-v2")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    policy = ContinualLearningPolicy(
        min_gold_records=args.min_gold_records,
        min_recent_gold_records=args.min_recent_gold_records,
        min_gold_domains=args.min_gold_domains,
        min_novelty_fraction=args.min_novelty_fraction,
        min_failure_clusters=args.min_failure_clusters,
        max_replay_records=args.max_replay_records,
        recent_window_size=args.recent_window_size,
        historical_window_size=args.historical_window_size,
        historical_fraction=args.historical_fraction,
        max_domain_fraction=args.max_domain_fraction,
        replay_seed=args.replay_seed,
        curriculum_seed=args.curriculum_seed,
    )
    manifest = ContinualLearningCycle(
        trajectories_path=args.trajectories,
        corrections_path=args.corrections,
        reviews_path=args.reviews,
        context_path=args.context,
        output_root=args.output_root,
        policy=policy,
    ).run()

    if args.json:
        print(json.dumps(manifest.model_dump(mode="json", by_alias=True), ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print("Phase 5 Contextual Continual Learning Cycle")
        print("===========================================")
        print(f"Cycle: {manifest.cycle_id}")
        print(f"Trajectories: {manifest.trajectory_count}")
        print(f"Context records: {manifest.context_record_count}")
        print(f"Contextualized trajectories: {manifest.contextualized_trajectory_count}")
        print(f"Tiers: {manifest.tier_counts}")
        print(f"Domains: {manifest.selected_domain_counts}")
        print(f"Difficulty: {manifest.curriculum_difficulty_counts}")
        print(f"Failure clusters: {manifest.failure_cluster_counts}")
        print(f"DPO candidates: {manifest.dpo_candidate_count}")
        print(f"SFT candidates: {manifest.sft_candidate_count}")
        print(f"Train trigger ready: {manifest.train_ready}")
        if manifest.train_block_reasons:
            print("Blocked by: " + ", ".join(manifest.train_block_reasons))
        print(f"Next: {manifest.recommended_action}")
        print("Training execution: DISABLED")
        print("Automatic promotion: DISABLED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
