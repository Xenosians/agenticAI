from pathlib import Path

from learning.continuous.store import (
    ContinuousLearningStore,
)
from learning.continuous.types import (
    AdaptiveRecipe,
    ContinuousCyclePlan,
    ContinuousCycleResult,
    LearningEvent,
)


def _event(event_id: str) -> LearningEvent:
    return LearningEvent(
        event_id=event_id,
        created_at="2026-09-28T00:00:00+00:00",
        trajectory_id="trajectory-" + event_id,
        model_key="hub-main",
    )


def _plan(cycle_id: str) -> ContinuousCyclePlan:
    return ContinuousCyclePlan(
        cycle_id=cycle_id,
        created_at="2026-09-28T00:00:00+00:00",
        event_ids=[],
        recipe=AdaptiveRecipe(),
    )


def test_restart_quarantines_unknown_claim(tmp_path: Path):
    store = ContinuousLearningStore(tmp_path / "state.sqlite3")
    store.initialize()
    assert store.enqueue_event(_event("unknown"))
    assert len(
        store.claim_events(
            cycle_id="cycle-unknown",
            limit=1,
        )
    ) == 1
    store.save_cycle_plan(_plan("cycle-unknown"))

    recovery = store.recover_interrupted_claims()
    assert recovery["quarantined_unknown"] == 1
    assert store.event_state_counts().get("failed") == 1
    latest = store.latest_cycle()
    assert latest is not None
    assert latest["state"] == "interrupted"


def test_restart_releases_proven_no_training_claim(tmp_path: Path):
    store = ContinuousLearningStore(tmp_path / "state.sqlite3")
    store.initialize()
    assert store.enqueue_event(_event("blocked"))
    store.claim_events(
        cycle_id="cycle-blocked",
        limit=1,
    )
    store.save_cycle_plan(_plan("cycle-blocked"))
    store.save_cycle_result(
        ContinuousCycleResult(
            cycle_id="cycle-blocked",
            completed_at="2026-09-28T00:01:00+00:00",
            state="blocked",
            training_started=False,
            training_executed=False,
            guard_signal="training_disabled",
        )
    )

    recovery = store.recover_interrupted_claims()
    assert recovery["released_without_training"] == 1
    assert store.pending_event_counts()["total"] == 1


def test_restart_consumes_proven_candidate_claim(tmp_path: Path):
    store = ContinuousLearningStore(tmp_path / "state.sqlite3")
    store.initialize()
    assert store.enqueue_event(_event("trained"))
    store.claim_events(
        cycle_id="cycle-trained",
        limit=1,
    )
    store.save_cycle_plan(_plan("cycle-trained"))
    store.save_cycle_result(
        ContinuousCycleResult(
            cycle_id="cycle-trained",
            completed_at="2026-09-28T00:01:00+00:00",
            state="candidate",
            training_started=True,
            training_executed=True,
            candidate_checkpoint_id="checkpoint-test",
        )
    )

    recovery = store.recover_interrupted_claims()
    assert recovery["completed_reconciled"] == 1
    assert store.pending_event_counts()["total"] == 0
    assert store.event_state_counts().get("consumed") == 1
