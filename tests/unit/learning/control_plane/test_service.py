"""The service layer: sessions, lease offers and message handling without a socket."""
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from learning.control_plane import canonical, protocol
from learning.control_plane.jobs import JobConflict, JobStore
from learning.control_plane.service import ControlPlaneService

START = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


class Wire:
    """A worker connection's outbound frames, parsed back through the same contract."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def __call__(self, text: str) -> None:
        self.texts.append(text)

    def take(self) -> list[protocol.Message]:
        messages = [protocol.parse_message(text) for text in self.texts]
        self.texts.clear()
        return messages


@pytest.fixture()
def store(tmp_path: Path) -> JobStore:
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.initialize()
    return store


def enqueue(store: JobStore, job_id: str = "job-1", key: str = "key-1") -> None:
    store.enqueue(job_id=job_id, idempotency_key=key, kind="specialist",
                  payload={"dataset": job_id}, now=START)


def started(lease: protocol.LeaseMessage, **overrides) -> protocol.StartedMessage:
    fields = {"sender": "worker-a", "job_id": lease.job_id,
              "lease_token": lease.lease_token, "attempt": lease.attempt}
    fields.update(overrides)
    return protocol.StartedMessage.create(**fields)


def progress(lease: protocol.LeaseMessage, **overrides) -> protocol.ProgressMessage:
    fields = {"sender": "worker-a", "job_id": lease.job_id,
              "lease_token": lease.lease_token, "attempt": lease.attempt,
              "stage": "sft", "fraction": 0.5}
    fields.update(overrides)
    return protocol.ProgressMessage.create(**fields)


def heartbeat(lease: protocol.LeaseMessage) -> protocol.HeartbeatMessage:
    return protocol.HeartbeatMessage.create(
        sender="worker-a", job_id=lease.job_id, lease_token=lease.lease_token,
        attempt=lease.attempt)


def succeeded(lease: protocol.LeaseMessage, result: dict | None = None) -> protocol.SucceededMessage:
    return protocol.SucceededMessage.create(
        sender="worker-a", job_id=lease.job_id, lease_token=lease.lease_token,
        attempt=lease.attempt, result={"adapter": "a"} if result is None else result)


def test_attach_sends_the_oldest_job_and_tracks_its_lease(store: JobStore):
    enqueue(store, "job-a", "key-a")
    enqueue(store, "job-b", "key-b")

    async def scenario():
        service = ControlPlaneService(store, clock=lambda: START)
        wire = Wire()
        session = await service.attach("worker-a", wire)

        [lease] = wire.take()
        assert isinstance(lease, protocol.LeaseMessage)
        assert lease.job_id == "job-a"
        assert lease.worker_id == "worker-a"
        assert lease.payload == {"dataset": "job-a"}
        assert session.active.job_id == "job-a"
        assert store.get("job-a").state == "leased"

    asyncio.run(scenario())


def test_attach_with_no_work_is_idle_until_notify(store: JobStore):
    async def scenario():
        service = ControlPlaneService(store, clock=lambda: START)
        wire = Wire()
        session = await service.attach("worker-a", wire)
        assert wire.take() == []
        assert session.active is None

        enqueue(store)
        await service.notify()

        [lease] = wire.take()
        assert isinstance(lease, protocol.LeaseMessage)
        assert lease.job_id == "job-1"

    asyncio.run(scenario())


def test_start_progress_heartbeat_and_success_move_the_job_and_offer_the_next(store: JobStore):
    enqueue(store, "job-1", "key-1")
    enqueue(store, "job-2", "key-2")
    now = [START]

    async def scenario():
        service = ControlPlaneService(store, clock=lambda: now[0])
        wire = Wire()
        session = await service.attach("worker-a", wire)
        [lease] = wire.take()

        assert await service.handle(session, protocol.message_json(started(lease))) is None
        assert store.get("job-1").state == "running"

        assert await service.handle(session, protocol.message_json(progress(lease))) is None
        assert store.get("job-1").progress == {"stage": "sft", "fraction": 0.5, "detail": None}

        now[0] = START + timedelta(seconds=10)
        renewed = await service.handle(session, protocol.message_json(heartbeat(lease)))
        assert isinstance(renewed, protocol.RenewedMessage)
        assert renewed.expires_at == canonical.iso(START + timedelta(seconds=310))
        assert wire.take() == [renewed]

        assert await service.handle(session, protocol.message_json(succeeded(lease))) is None
        assert store.get("job-1").state == "succeeded"
        assert store.get("job-1").result == {"adapter": "a"}

        [next_lease] = wire.take()
        assert isinstance(next_lease, protocol.LeaseMessage)
        assert next_lease.job_id == "job-2"
        assert session.active.job_id == "job-2"

    asyncio.run(scenario())


def test_progress_before_start_is_refused_and_the_lease_is_not_used(store: JobStore):
    enqueue(store)

    async def scenario():
        service = ControlPlaneService(store, clock=lambda: START)
        wire = Wire()
        session = await service.attach("worker-a", wire)
        [lease] = wire.take()

        lost = await service.handle(session, protocol.message_json(progress(lease)))

        assert isinstance(lost, protocol.LeaseLostMessage)
        assert lost.reason_code == "job_conflict"
        assert store.get("job-1").state == "leased"
        assert session.active is None
        assert wire.take() == [lost]

    asyncio.run(scenario())


def test_a_message_for_another_attempt_is_fenced(store: JobStore):
    enqueue(store)

    async def scenario():
        service = ControlPlaneService(store, clock=lambda: START)
        wire = Wire()
        session = await service.attach("worker-a", wire)
        [lease] = wire.take()

        lost = await service.handle(
            session, protocol.message_json(started(lease, attempt=lease.attempt + 1)))

        assert isinstance(lost, protocol.LeaseLostMessage)
        assert lost.reason_code == "lease_lost"
        assert store.get("job-1").state == "leased"
        assert session.active is None

    asyncio.run(scenario())


def test_a_message_from_another_sender_is_refused_before_the_store(store: JobStore):
    enqueue(store)

    async def scenario():
        service = ControlPlaneService(store, clock=lambda: START)
        wire = Wire()
        session = await service.attach("worker-a", wire)
        [lease] = wire.take()

        with pytest.raises(canonical.ContractError):
            await service.handle(session, protocol.message_json(started(lease, sender="worker-b")))

        assert store.get("job-1").state == "leased"
        assert wire.take() == []

    asyncio.run(scenario())


def test_a_stale_token_cannot_change_a_job_leased_to_another_worker(store: JobStore):
    enqueue(store)

    async def scenario():
        service = ControlPlaneService(store, clock=lambda: START)
        wire = Wire()
        session = await service.attach("worker-a", wire)
        [first] = wire.take()

        store.reconcile_expired(now=START + timedelta(seconds=301))
        store.resolve("job-1", outcome="retry", now=START + timedelta(seconds=301))
        second = store.lease(worker_id="worker-b", now=START + timedelta(seconds=302))
        assert second is not None

        lost = await service.handle(session, protocol.message_json(started(first)))

        assert isinstance(lost, protocol.LeaseLostMessage)
        assert store.get("job-1").worker_id == "worker-b"
        assert store.get("job-1").state == "leased"

    asyncio.run(scenario())


def test_a_reconciliation_report_is_evidence_and_the_operator_still_resolves(store: JobStore):
    enqueue(store)
    now = [START]

    async def scenario():
        service = ControlPlaneService(store, clock=lambda: now[0])
        wire = Wire()
        session = await service.attach("worker-a", wire)
        wire.take()

        now[0] = START + timedelta(seconds=301)
        store.reconcile_expired(now=now[0])
        assert store.get("job-1").state == "needs_reconcile"

        request = await service.request_reconciliation("job-1")
        assert isinstance(request, protocol.ReconcileRequestMessage)
        assert wire.take() == [request]

        report = protocol.ReconcileResultMessage.create(
            sender="worker-a", job_id="job-1", outcome="succeeded",
            result={"adapter": "a"})
        assert await service.handle(session, protocol.message_json(report)) is None

        assert store.get("job-1").state == "needs_reconcile"
        assert [item.result for item in store.reconciliations("job-1")] == [{"adapter": "a"}]
        assert session.active is None

    asyncio.run(scenario())


def test_reconciliation_is_only_requested_for_an_uncertain_job(store: JobStore):
    enqueue(store)

    async def scenario():
        service = ControlPlaneService(store, clock=lambda: START)
        with pytest.raises(JobConflict):
            await service.request_reconciliation("job-1")

    asyncio.run(scenario())


def test_a_reconnect_adopts_the_held_job_and_does_not_offer_a_second_one(store: JobStore):
    enqueue(store, "job-1", "key-1")

    async def scenario():
        service = ControlPlaneService(store, clock=lambda: START)
        wire = Wire()
        first = await service.attach("worker-a", wire)
        [lease] = wire.take()
        await service.detach(first)

        enqueue(store, "job-2", "key-2")
        wire = Wire()
        session = await service.attach("worker-a", wire)

        assert wire.take() == []
        assert session.active.job_id == "job-1"
        assert store.get("job-2").state == "queued"

        assert await service.handle(session, protocol.message_json(started(lease))) is None
        assert store.get("job-1").state == "running"

        assert await service.handle(session, protocol.message_json(succeeded(lease))) is None
        [next_lease] = wire.take()
        assert isinstance(next_lease, protocol.LeaseMessage)
        assert next_lease.job_id == "job-2"

    asyncio.run(scenario())


def test_an_expired_lease_stops_being_current_and_a_new_job_is_offered(store: JobStore):
    enqueue(store, "job-1", "key-1")
    now = [START]

    async def scenario():
        service = ControlPlaneService(store, clock=lambda: now[0])
        wire = Wire()
        session = await service.attach("worker-a", wire)
        wire.take()

        now[0] = START + timedelta(seconds=301)
        enqueue(store, "job-2", "key-2")
        await service.notify()

        assert store.get("job-1").state == "needs_reconcile"
        [lease] = wire.take()
        assert isinstance(lease, protocol.LeaseMessage)
        assert lease.job_id == "job-2"
        assert session.active.job_id == "job-2"

    asyncio.run(scenario())


def test_a_retryable_failure_is_offered_again_as_a_new_attempt(store: JobStore):
    enqueue(store)

    async def scenario():
        service = ControlPlaneService(store, clock=lambda: START)
        wire = Wire()
        session = await service.attach("worker-a", wire)
        [lease] = wire.take()
        await service.handle(session, protocol.message_json(started(lease)))

        failed = protocol.FailedMessage.create(
            sender="worker-a", job_id=lease.job_id, lease_token=lease.lease_token,
            attempt=lease.attempt, reason_code="cuda_oom", retryable=True)
        assert await service.handle(session, protocol.message_json(failed)) is None

        [retry] = wire.take()
        assert isinstance(retry, protocol.LeaseMessage)
        assert retry.job_id == "job-1"
        assert retry.attempt == 2
        assert retry.lease_token != lease.lease_token
        assert store.get("job-1").state == "leased"

    asyncio.run(scenario())
