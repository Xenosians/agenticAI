from __future__ import annotations

import argparse
import json

from config import Settings
from learning.continuous.config import (
    ContinuousLearningSettings,
)
from learning.continuous.store import (
    ContinuousLearningStore,
)
from learning.continuous.targets import (
    build_continuous_target_coverage,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect sanitized effective continuous-learning controls, "
            "durable queue state, restart state, and target coverage."
        )
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--recover-interrupted",
        action="store_true",
        help=(
            "Explicitly reconcile stale claims. Ambiguous optimizer "
            "outcomes are quarantined, never replayed."
        ),
    )
    return parser


def build_snapshot(
    *,
    recover_interrupted: bool = False,
) -> dict:
    continuous = ContinuousLearningSettings.from_env()
    app = Settings()
    store = ContinuousLearningStore(
        continuous.state_db_path
    )
    store.initialize()

    recovery = (
        store.recover_interrupted_claims()
        if recover_interrupted
        else None
    )
    pending = store.pending_event_counts()
    age = store.oldest_pending_event_age_seconds()
    trigger_ready = (
        pending["total"] >= continuous.min_events_per_cycle
        or pending["failures"]
        >= continuous.failure_events_trigger
        or (
            age is not None
            and age >= continuous.max_cycle_age_seconds
        )
    )

    coverage = build_continuous_target_coverage(
        hub_model_key=app.hub_model_key,
        agent_directory=app.agents_dir,
    )
    blockers = [
        "shared_model_role_eval_missing:" + item.target_component
        for item in coverage
        if (
            item.model_key == app.hub_model_key
            and item.target_component != "hub"
            and not item.auto_promotion_supported
        )
    ]

    return {
        "schema": "continuous-learning-status.v1",
        "enabled": continuous.enabled,
        "controls": {
            "auto_train": continuous.auto_train,
            "auto_evaluate": continuous.auto_evaluate,
            "ppo_enabled": continuous.ppo_enabled,
            "auto_promote": continuous.auto_promote,
        },
        "training_stage_ready": bool(
            continuous.enabled
            and continuous.auto_train
            and trigger_ready
        ),
        "thresholds": {
            "poll_seconds": continuous.poll_seconds,
            "idle_seconds_before_training": (
                continuous.idle_seconds_before_training
            ),
            "min_events_per_cycle": (
                continuous.min_events_per_cycle
            ),
            "max_events_per_cycle": (
                continuous.max_events_per_cycle
            ),
            "failure_events_trigger": (
                continuous.failure_events_trigger
            ),
            "max_cycle_age_seconds": (
                continuous.max_cycle_age_seconds
            ),
            "corpus_pages_per_cycle": (
                continuous.recipe.corpus_pages_per_cycle
            ),
            "replay_pages_per_cycle": (
                continuous.recipe.replay_pages_per_cycle
            ),
            "max_optimizer_steps": (
                continuous.recipe.max_optimizer_steps
            ),
            "ppo_updates": continuous.recipe.ppo_updates,
        },
        "pending_events": pending,
        "oldest_pending_event_age_seconds": age,
        "trigger_ready": trigger_ready,
        "event_states": store.event_state_counts(),
        "corpus_page_states": (
            store.corpus_page_state_counts()
        ),
        "training_cycle_count": store.training_cycle_count(),
        "latest_cycle": store.latest_cycle(),
        "target_coverage": [
            item.model_dump(mode="json")
            for item in coverage
        ],
        "shared_model_promotion_blockers": blockers,
        "recovery": recovery,
    }


def main() -> int:
    args = build_parser().parse_args()
    snapshot = build_snapshot(
        recover_interrupted=args.recover_interrupted
    )

    if args.json:
        print(
            json.dumps(
                snapshot,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0

    print("Continuous Learning Status")
    print("==========================")
    print("enabled=" + str(snapshot["enabled"]))
    print(
        "controls="
        + json.dumps(snapshot["controls"], sort_keys=True)
    )
    print(
        "pending_events="
        + json.dumps(
            snapshot["pending_events"],
            sort_keys=True,
        )
    )
    print("trigger_ready=" + str(snapshot["trigger_ready"]))
    print(
        "training_stage_ready="
        + str(snapshot["training_stage_ready"])
    )
    print(
        "training_cycle_count="
        + str(snapshot["training_cycle_count"])
    )
    print(
        "shared_model_promotion_blockers="
        + json.dumps(
            snapshot["shared_model_promotion_blockers"],
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
