"""Durable jobs: idempotent enqueue, fenced leases, expiry and explicit reconciliation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import threading

import pytest

from learning.control_plane import canonical
from learning.control_plane.jobs import (
    IdempotencyConflict,
    JobConflict,
    JobNotFound,
    JobStore,
    JobStoreError,
    LeaseLost,
)

START = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def store(tmp_path: Path) -> JobStore:
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.initialize()
    return store


def enqueue(store: JobStore, *, job_id: str = "job-1", key: str = "key-1",
            payload: dict | None = None, retry_safe: bool = False,
            max_attempts: int | None = None, now: datetime = START):
    return store.enqueue(
        job_id=job_id,
        idempotency_key=key,
        kind="specialist",
        payload={"dataset": "d1"} if payload is None else payload,
        retry_safe=retry_safe,
        max_attempts=max_attempts,
        now=now,
    )


def test_the_same_key_and_payload_returns_the_same_job(store: JobStore):
    first = enqueue(store)
    second = enqueue(store)

    assert first.created is True
    assert second.created is False
    assert second.job.job_id == first.job.job_id
    assert store.counts() == {"queued": 1}


def test_a_replayed_key_with_another_payload_is_a_conflict(store: JobStore):
    enqueue(store)

    with pytest.raises(IdempotencyConflict):
        enqueue(store, payload={"dataset": "d2"})
    with pytest.raises(IdempotencyConflict):
        enqueue(store, job_id="job-2")

    assert store.get("job-1").payload == {"dataset": "d1"}


def test_a_job_id_cannot_be_reused_under_another_key(store: JobStore):
    enqueue(store)

    with pytest.raises(JobConflict):
        enqueue(store, key="key-2")

    assert store.counts() == {"queued": 1}


@pytest.mark.parametrize(
    "values",
    [
        {"job_id": ""},
        {"job_id": "bad job"},
        {"key": ""},
        {"key": "bad key"},
        {"payload": ["not", "an", "object"]},
        {"payload": {"score": float("nan")}},
    ],
)
def test_an_invalid_job_is_refused_before_a_row_is_written(store: JobStore, values: dict):
    kwargs = {"job_id": "job-1", "key": "key-1", "payload": {"dataset": "d1"}}
    kwargs.update(values)

    with pytest.raises(canonical.ContractError):
        enqueue(store, **kwargs)

    assert store.counts() == {}


def test_lease_takes_the_oldest_queued_job_and_increments_its_attempt(store: JobStore):
    enqueue(store, job_id="job-a", key="key-a")
    enqueue(store, job_id="job-b", key="key-b", now=START + timedelta(seconds=1))

    lease = store.lease(worker_id="worker-a", lease_seconds=300, now=START)

    assert lease is not None
    assert lease.job_id == "job-a"
    assert lease.attempt == 1
    assert lease.job.state == "leased"
    assert lease.job.worker_id == "worker-a"
    assert lease.job.lease_token == lease.lease_token
    assert lease.expires_at == canonical.iso(START + timedelta(seconds=300))

    second = store.lease(worker_id="worker-a", now=START)
    assert second is not None and second.job_id == "job-b"


def test_lease_can_be_restricted_to_one_kind(store: JobStore):
    store.enqueue(job_id="job-hub", idempotency_key="hub", kind="hub", payload={}, now=START)
    store.enqueue(job_id="job-specialist", idempotency_key="specialist", kind="specialist",
                  payload={}, now=START + timedelta(seconds=1))

    lease = store.lease(worker_id="worker-a", kinds=["specialist"], now=START)

    assert lease is not None and lease.job_id == "job-specialist"
    assert store.get("job-hub").state == "queued"


def test_lease_returns_none_when_nothing_is_queued(store: JobStore):
    assert store.lease(worker_id="worker-a", now=START) is None


def test_an_empty_kind_filter_leases_nothing(store: JobStore):
    enqueue(store, payload={})

    assert store.lease(worker_id="worker-a", kinds=[], now=START) is None
    assert store.get("job-1").state == "queued"


def test_progress_waits_for_the_workers_own_start(store: JobStore):
    enqueue(store, payload={})
    lease = store.lease(worker_id="worker-a", now=START)
    assert lease is not None

    with pytest.raises(JobConflict):
        store.progress(lease.job_id, lease.lease_token, stage="sft", fraction=0.5, now=START)

    started = store.start(lease.job_id, lease.lease_token, now=START)
    assert started.state == "running"
    assert store.start(lease.job_id, lease.lease_token, now=START).state == "running"

    progress = store.progress(lease.job_id, lease.lease_token, stage="sft", fraction=0.25,
                              detail="  step   one  ", now=START + timedelta(seconds=1))
    assert progress.progress == {"stage": "sft", "fraction": 0.25, "detail": "step one"}
    assert progress.heartbeat_at == canonical.iso(START + timedelta(seconds=1))


@pytest.mark.parametrize("fraction", [-0.1, 1.1, True, "0.5"])
def test_progress_refuses_a_fraction_that_is_not_between_zero_and_one(store: JobStore,
                                                                     fraction):
    enqueue(store, payload={})
    lease = store.lease(worker_id="worker-a", now=START)
    assert lease is not None
    store.start(lease.job_id, lease.lease_token, now=START)

    with pytest.raises(canonical.ContractError):
        store.progress(lease.job_id, lease.lease_token, stage="sft", fraction=fraction,
                       now=START)


def test_a_stale_token_cannot_touch_a_job_that_was_leased_again(store: JobStore):
    enqueue(store, payload={})
    first = store.lease(worker_id="worker-a", now=START)
    assert first is not None
    store.start(first.job_id, first.lease_token, now=START)

    expired = store.reconcile_expired(now=START + timedelta(seconds=301))
    assert [job.state for job in expired] == ["needs_reconcile"]

    store.resolve(first.job_id, outcome="retry", now=START + timedelta(seconds=301))
    second = store.lease(worker_id="worker-b", now=START + timedelta(seconds=302))
    assert second is not None and second.lease_token != first.lease_token

    for operation in (
        lambda: store.start(first.job_id, first.lease_token, now=START + timedelta(seconds=303)),
        lambda: store.progress(first.job_id, first.lease_token, stage="sft", fraction=0.5,
                               now=START + timedelta(seconds=303)),
        lambda: store.heartbeat(first.job_id, first.lease_token, now=START + timedelta(seconds=303)),
        lambda: store.succeed(first.job_id, first.lease_token, result={"ok": True},
                              now=START + timedelta(seconds=303)),
        lambda: store.fail(first.job_id, first.lease_token, reason_code="crashed",
                           now=START + timedelta(seconds=303)),
    ):
        with pytest.raises(LeaseLost):
            operation()

    assert store.get(first.job_id).state == "leased"
    assert store.get(first.job_id).worker_id == "worker-b"


def test_heartbeat_only_extends_an_unexpired_lease(store: JobStore):
    enqueue(store, payload={})
    lease = store.lease(worker_id="worker-a", lease_seconds=300, now=START)
    assert lease is not None

    renewed = store.heartbeat(lease.job_id, lease.lease_token, extend_seconds=300,
                              now=START + timedelta(seconds=100))
    assert renewed.lease_expires_at == canonical.iso(START + timedelta(seconds=400))

    with pytest.raises(LeaseLost):
        store.heartbeat(lease.job_id, lease.lease_token, now=START + timedelta(seconds=401))


def test_expiry_is_exclusive_at_the_deadline(store: JobStore):
    enqueue(store, payload={}, retry_safe=False)
    lease = store.lease(worker_id="worker-a", lease_seconds=300, now=START)
    assert lease is not None

    assert store.reconcile_expired(now=START + timedelta(seconds=299)) == []
    expired = store.reconcile_expired(now=START + timedelta(seconds=300))

    assert [job.state for job in expired] == ["needs_reconcile"]
    assert expired[0].reason_code == "lease_expired"


def test_a_retry_safe_expiry_requeues_until_the_attempts_run_out(store: JobStore):
    enqueue(store, payload={}, retry_safe=True, max_attempts=2)
    first = store.lease(worker_id="worker-a", now=START)
    assert first is not None

    store.reconcile_expired(now=START + timedelta(seconds=301))
    requeued = store.get(first.job_id)
    assert requeued.state == "queued"
    assert requeued.worker_id == "worker-a"
    assert requeued.lease_token is None
    assert requeued.reason_code == "lease_expired"

    second = store.lease(worker_id="worker-b", now=START + timedelta(seconds=302))
    assert second is not None and second.attempt == 2

    store.reconcile_expired(now=START + timedelta(seconds=700))
    exhausted = store.get(first.job_id)
    assert exhausted.state == "needs_reconcile"
    assert exhausted.worker_id == "worker-b"
    assert store.lease(worker_id="worker-c", now=START + timedelta(seconds=701)) is None


def test_an_expired_job_that_is_not_retry_safe_waits_for_an_operator(store: JobStore):
    enqueue(store, payload={}, retry_safe=False)
    lease = store.lease(worker_id="worker-a", now=START)
    assert lease is not None

    store.reconcile_expired(now=START + timedelta(seconds=301))

    assert store.get(lease.job_id).state == "needs_reconcile"
    assert store.lease(worker_id="worker-b", now=START + timedelta(seconds=302)) is None


def test_resolve_refuses_a_job_that_is_not_needs_reconcile(store: JobStore):
    enqueue(store, payload={}, retry_safe=False)

    with pytest.raises(JobConflict):
        store.resolve("job-1", outcome="retry", reason_code="operator_retry", now=START)

    lease = store.lease(worker_id="worker-a", now=START)
    assert lease is not None
    store.reconcile_expired(now=START + timedelta(seconds=301))

    resolved = store.resolve("job-1", outcome="retry", reason_code="operator_retry",
                             now=START + timedelta(seconds=302))
    assert resolved.state == "queued"


def test_resolve_retry_requeues_a_reconciled_job(store: JobStore):
    enqueue(store, payload={}, retry_safe=False, max_attempts=2)
    lease = store.lease(worker_id="worker-a", now=START)
    assert lease is not None
    store.reconcile_expired(now=START + timedelta(seconds=301))

    resolved = store.resolve(lease.job_id, outcome="retry", reason_code="operator_retry",
                             detail="the worker was rebooted", now=START + timedelta(seconds=302))
    assert resolved.state == "queued"
    assert resolved.reason_code == "operator_retry"
    assert resolved.reconciled_at == canonical.iso(START + timedelta(seconds=302))

    again = store.lease(worker_id="worker-b", now=START + timedelta(seconds=303))
    assert again is not None and again.attempt == 2


def test_resolve_can_record_a_result_an_operator_observed(store: JobStore):
    enqueue(store, payload={}, retry_safe=False)
    store.lease(worker_id="worker-a", now=START)
    store.reconcile_expired(now=START + timedelta(seconds=301))

    resolved = store.resolve("job-1", outcome="succeeded", result={"adapter": "a"},
                             now=START + timedelta(seconds=302))

    assert resolved.state == "succeeded"
    assert resolved.result == {"adapter": "a"}
    assert resolved.result_sha256 == canonical.digest({"adapter": "a"})


def test_resolve_failed_and_cancelled_are_terminal(store: JobStore):
    for job_id, key in (("job-failed", "key-failed"), ("job-cancelled", "key-cancelled")):
        enqueue(store, job_id=job_id, key=key, payload={}, retry_safe=False)
        store.lease(worker_id="worker-a", now=START)
        store.reconcile_expired(now=START + timedelta(seconds=301))
        outcome = "failed" if job_id == "job-failed" else "cancelled"
        resolved = store.resolve(job_id, outcome=outcome, reason_code="operator_decision",
                                 now=START + timedelta(seconds=302))
        assert resolved.state == outcome

    with pytest.raises(JobConflict):
        store.resolve("job-failed", outcome="retry", now=START + timedelta(seconds=303))


def test_a_result_replayed_with_the_same_digest_is_a_no_op(store: JobStore):
    enqueue(store, payload={})
    lease = store.lease(worker_id="worker-a", now=START)
    assert lease is not None
    store.start(lease.job_id, lease.lease_token, now=START)

    first = store.succeed(lease.job_id, lease.lease_token, result={"adapter": "a"}, now=START)
    replay = store.succeed(lease.job_id, "a-different-token", result={"adapter": "a"},
                           now=START + timedelta(seconds=1))

    assert first.state == "succeeded" and replay.state == "succeeded"
    assert first.result_sha256 == replay.result_sha256 == canonical.digest({"adapter": "a"})
    assert replay.updated_at == first.updated_at

    with pytest.raises(JobConflict):
        store.succeed(lease.job_id, lease.lease_token, result={"adapter": "b"}, now=START)


def test_failure_retries_only_when_the_worker_says_it_is_safe_and_an_attempt_remains(
        store: JobStore):
    enqueue(store, payload={}, max_attempts=2)
    first = store.lease(worker_id="worker-a", now=START)
    assert first is not None
    store.start(first.job_id, first.lease_token, now=START)

    retried = store.fail(first.job_id, first.lease_token, reason_code="cuda_oom",
                         detail="out of memory", retryable=True, now=START)
    assert retried.state == "queued"

    second = store.lease(worker_id="worker-a", now=START + timedelta(seconds=1))
    assert second is not None and second.attempt == 2
    store.start(second.job_id, second.lease_token, now=START + timedelta(seconds=1))

    exhausted = store.fail(second.job_id, second.lease_token, reason_code="cuda_oom",
                           retryable=True, now=START + timedelta(seconds=2))
    assert exhausted.state == "failed"
    assert exhausted.detail is None


def test_a_failure_the_worker_calls_unsafe_is_terminal_at_once(store: JobStore):
    enqueue(store, payload={})
    lease = store.lease(worker_id="worker-a", now=START)
    assert lease is not None
    store.start(lease.job_id, lease.lease_token, now=START)

    failed = store.fail(lease.job_id, lease.lease_token, reason_code="bad_dataset",
                        retryable=False, now=START)

    assert failed.state == "failed"
    assert failed.reason_code == "bad_dataset"
    assert store.lease(worker_id="worker-b", now=START) is None


def test_cancel_ends_a_live_job_and_a_late_result_cannot_change_it(store: JobStore):
    enqueue(store, payload={})
    lease = store.lease(worker_id="worker-a", now=START)
    assert lease is not None

    cancelled = store.cancel(lease.job_id, reason_code="owner_cancelled", now=START)
    assert cancelled.state == "cancelled"
    assert cancelled.lease_token is None

    with pytest.raises(JobConflict):
        store.succeed(lease.job_id, lease.lease_token, result={"adapter": "a"}, now=START)

    assert store.cancel(lease.job_id, now=START).state == "cancelled"


def test_a_terminal_job_is_never_leased_again(store: JobStore):
    enqueue(store, payload={})
    lease = store.lease(worker_id="worker-a", now=START)
    assert lease is not None
    store.start(lease.job_id, lease.lease_token, now=START)
    store.succeed(lease.job_id, lease.lease_token, result={"ok": True}, now=START)

    assert store.lease(worker_id="worker-b", now=START) is None
    assert store.list(state="succeeded")[0].job_id == lease.job_id


def test_claim_message_is_true_once_and_false_every_replay(store: JobStore):
    assert store.claim_message("message-123", sender="worker-a", now=START) is True
    assert store.claim_message("message-123", sender="worker-a", now=START) is False

    with pytest.raises(canonical.ContractError):
        store.claim_message("short")


def test_pruning_the_replay_guard_keeps_only_the_recent_window(store: JobStore):
    assert store.claim_message("message-old", sender="worker-a", now=START) is True
    assert store.claim_message("message-new", sender="worker-a",
                               now=START + timedelta(days=10)) is True

    removed = store.prune_messages(older_than_seconds=5 * 24 * 3600,
                                   now=START + timedelta(days=12))

    assert removed == 1
    assert store.claim_message("message-old", sender="worker-a", now=START) is True
    assert store.claim_message("message-new", sender="worker-a", now=START) is False


def test_a_database_from_a_newer_control_plane_is_refused(tmp_path: Path):
    path = tmp_path / "jobs.sqlite3"
    JobStore(path).initialize()
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 99")
        conn.commit()

    with pytest.raises(JobStoreError):
        JobStore(path).initialize()


def test_missing_jobs_are_reported_as_not_found(store: JobStore):
    with pytest.raises(JobNotFound):
        store.get("job-missing")
    assert store.find("job-missing") is None
    with pytest.raises(JobNotFound):
        store.cancel("job-missing", now=START)


def test_two_store_instances_never_lease_one_job_twice(tmp_path: Path):
    path = tmp_path / "jobs.sqlite3"
    JobStore(path).initialize()
    JobStore(path).enqueue(job_id="job-1", idempotency_key="key-1", kind="specialist",
                           payload={}, now=START)
    barrier = threading.Barrier(8)

    def attempt(_: int):
        store = JobStore(path)
        barrier.wait(timeout=10)
        return store.lease(worker_id="worker-a", now=START)

    with ThreadPoolExecutor(max_workers=8) as pool:
        leases = list(pool.map(attempt, range(8)))

    won = [lease for lease in leases if lease is not None]
    assert len(won) == 1
    assert JobStore(path).get("job-1").attempts == 1
