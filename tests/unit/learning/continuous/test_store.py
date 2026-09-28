from pathlib import Path

from learning.continuous.store import (
    ContinuousLearningStore,
)
from learning.continuous.types import (
    LearningEvent,
)


def _event(
    event_id: str,
    *,
    had_error: bool = False,
) -> LearningEvent:
    return LearningEvent(
        event_id=event_id,
        created_at="2026-09-28T00:00:00+00:00",
        trajectory_id=(
            "trajectory-"
            + event_id
        ),
        model_key="hub-main",
        had_error=had_error,
    )


def test_event_queue_is_durable_and_deduplicated(
    tmp_path: Path,
):
    store = ContinuousLearningStore(
        tmp_path
        / "state.sqlite3"
    )

    store.initialize()

    assert store.enqueue_event(
        _event(
            "a",
            had_error=True,
        )
    )

    assert not store.enqueue_event(
        _event(
            "a",
            had_error=True,
        )
    )

    assert store.enqueue_event(
        _event(
            "b"
        )
    )

    counts = (
        store
        .pending_event_counts()
    )

    assert counts == {
        "total": 2,
        "failures": 1,
    }

    claimed = store.claim_events(
        cycle_id="cycle-1",
        limit=8,
    )

    assert [
        item.event_id
        for item in claimed
    ] == [
        "a",
        "b",
    ]

    assert (
        store
        .pending_event_counts()[
            "total"
        ]
        == 0
    )

    store.finalize_events(
        cycle_id="cycle-1",
        consumed=False,
    )

    assert (
        store
        .pending_event_counts()[
            "total"
        ]
        == 2
    )
