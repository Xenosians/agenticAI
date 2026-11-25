"""Connect validated socket messages to the durable store, one session per worker.

The service is deliberately transport-free: a session is an authenticated worker id plus an
``await send(text)`` callable. The WebSocket server only authenticates the connection and pumps
frames; this module decides which store transition a message may cause.

Rules kept here, not in the socket:

- a message sender must match the authenticated session; a connection cannot claim another worker;
- an echoed attempt must match the job's current attempt, and the store's lease token is the fence;
- a successful or failed transition clears the session and immediately offers the next job;
- ``reconcile_result`` is stored as evidence and never changes job state; an operator resolves it;
- a transport send is a notification. It never acknowledges durable completion on its own.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Union

from learning.control_plane import canonical
from learning.control_plane.jobs import (
    ACTIVE_STATES,
    JobConflict,
    JobNotFound,
    JobRecord,
    JobStore,
    LeaseLost,
)
from learning.control_plane.protocol import (
    FailedMessage,
    HeartbeatMessage,
    LeaseLostMessage,
    LeaseMessage,
    MessageType,
    ProgressMessage,
    ReconcileRequestMessage,
    ReconcileResultMessage,
    RenewedMessage,
    StartedMessage,
    SucceededMessage,
    message_json,
    parse_inbound,
)

Send = Callable[[str], Awaitable[None]]
Clock = Callable[[], datetime]
JobMessage = Union[StartedMessage, ProgressMessage, SucceededMessage, FailedMessage,
                   HeartbeatMessage]


class _Lost(Exception):
    """A fenced transition failed; the worker is told to stop and the row is unchanged."""

    def __init__(self, message: LeaseLostMessage) -> None:
        super().__init__(message.detail or message.reason_code)
        self.message = message


@dataclass(frozen=True)
class ActiveJob:
    job_id: str
    lease_token: str
    attempt: int
    kind: str
    expires_at: str


@dataclass
class WorkerSession:
    worker_id: str
    kinds: tuple[str, ...]
    send: Send
    active: ActiveJob | None = None

    def track(self, record: JobRecord) -> ActiveJob:
        """Remember the job this session now owns; a session never tracks two at once."""
        if record.lease_token is None or record.lease_expires_at is None:
            raise canonical.ContractError("an active job must carry its lease")
        if self.active is not None and self.active.job_id != record.job_id:
            raise canonical.ContractError("a worker session may hold one job at a time")
        self.active = ActiveJob(
            job_id=record.job_id,
            lease_token=record.lease_token,
            attempt=record.attempts,
            kind=record.kind,
            expires_at=record.lease_expires_at,
        )
        return self.active

    def finish(self, job_id: str) -> None:
        if self.active is not None and self.active.job_id == job_id:
            self.active = None


class ControlPlaneService:
    """One control plane, many worker sessions, one lease per job at a time."""

    def __init__(self, store: JobStore, *, sender: str = "janus-control",
                 lease_seconds: float = 300.0, clock: Clock = canonical.utc_now) -> None:
        canonical.require(canonical.SENDER, sender, "control-plane sender")
        if lease_seconds <= 0:
            raise canonical.ContractError("lease_seconds must be positive")
        self.store = store
        self.sender = sender
        self.lease_seconds = float(lease_seconds)
        self.clock = clock
        self._sessions: dict[str, WorkerSession] = {}
        self._lock = asyncio.Lock()

    def now(self) -> datetime:
        return self.clock()

    def sessions(self) -> tuple[WorkerSession, ...]:
        return tuple(self._sessions.values())

    async def attach(self, worker_id: str, send: Send, *,
                     kinds: Iterable[str] | None = None) -> WorkerSession:
        """Authenticate one worker session and give it its job, or hold it ready."""
        canonical.require(canonical.WORKER_ID, worker_id, "worker id")
        selected = () if kinds is None else tuple(
            canonical.require(canonical.KIND, kind, "job kind") for kind in kinds
        )
        async with self._lock:
            if worker_id in self._sessions:
                raise canonical.ContractError(f"worker {worker_id!r} is already attached")
            session = WorkerSession(worker_id=worker_id, kinds=selected, send=send)
            self._sessions[worker_id] = session
            try:
                self.store.reconcile_expired(now=self.now())
                held = self.store.held_by(worker_id)
                if held:
                    session.track(held[0])
                else:
                    await self._offer(session)
            except Exception:
                self._sessions.pop(worker_id, None)
                session.active = None
                raise
            return session

    async def detach(self, session: WorkerSession) -> None:
        """Forget the connection but keep the lease; an unreaped job expires normally."""
        async with self._lock:
            if self._sessions.get(session.worker_id) is session:
                del self._sessions[session.worker_id]
            session.active = None

    async def notify(self) -> None:
        """Offer work to every session whose tracked job is no longer held."""
        async with self._lock:
            for session in tuple(self._sessions.values()):
                await self._offer(session)

    async def handle(self, session: WorkerSession, text: str | bytes) -> MessageType | None:
        """Apply one inbound frame and send any reply. A malformed frame raises ContractError."""
        message = parse_inbound(text)
        if message.sender != session.worker_id:
            raise canonical.ContractError(
                "a control-plane message must carry the authenticated worker as its sender"
            )
        async with self._lock:
            if self._sessions.get(session.worker_id) is not session:
                raise canonical.ContractError("the worker session is no longer attached")
            try:
                return await self._dispatch(session, message)
            except _Lost as lost:
                await session.send(message_json(lost.message))
                await self._offer(session)
                return lost.message

    async def request_reconciliation(self, job_id: str, *,
                                     reason_code: str | None = None) -> ReconcileRequestMessage | None:
        """Ask the worker that last held an uncertain job what it observed, if it is online."""
        async with self._lock:
            job = self.store.get(job_id)
            if job.state != "needs_reconcile":
                raise JobConflict(f"job {job_id!r} is {job.state}, not needs_reconcile")
            if job.worker_id is None:
                return None
            session = self._sessions.get(job.worker_id)
            if session is None:
                return None
            message = ReconcileRequestMessage.create(
                sender=self.sender,
                job_id=job_id,
                reason_code=reason_code or job.reason_code or "lease_expired",
                detail=job.detail,
            )
            await session.send(message_json(message))
            return message

    # -- dispatch --------------------------------------------------------------------------------------

    async def _dispatch(self, session: WorkerSession, message: MessageType) -> MessageType | None:
        if isinstance(message, ReconcileResultMessage):
            return await self._reconcile(session, message)
        if isinstance(message, StartedMessage):
            record = self._transition(
                session, message, lambda: self.store.start(
                    message.job_id, message.lease_token, now=self.now()))
            session.track(record)
            self._claim(session, message)
            return None
        if isinstance(message, ProgressMessage):
            record = self._transition(
                session, message, lambda: self.store.progress(
                    message.job_id, message.lease_token, stage=message.stage,
                    fraction=message.fraction, detail=message.detail, now=self.now()))
            session.track(record)
            self._claim(session, message)
            return None
        if isinstance(message, HeartbeatMessage):
            record = self._transition(
                session, message, lambda: self.store.heartbeat(
                    message.job_id, message.lease_token, extend_seconds=self.lease_seconds,
                    now=self.now()))
            session.track(record)
            if record.lease_expires_at is None:
                raise canonical.ContractError("a renewed lease must have an expiry")
            renewed = RenewedMessage.create(
                sender=self.sender,
                job_id=message.job_id,
                lease_token=message.lease_token,
                attempt=message.attempt,
                expires_at=record.lease_expires_at,
            )
            self._claim(session, message)
            await session.send(message_json(renewed))
            return renewed
        if isinstance(message, SucceededMessage):
            self._transition(
                session, message, lambda: self.store.succeed(
                    message.job_id, message.lease_token, result=message.result, now=self.now()))
            session.finish(message.job_id)
            self._claim(session, message)
            await self._offer(session)
            return None
        if isinstance(message, FailedMessage):
            self._transition(
                session, message, lambda: self.store.fail(
                    message.job_id, message.lease_token, reason_code=message.reason_code,
                    detail=message.detail, retryable=message.retryable, now=self.now()))
            session.finish(message.job_id)
            self._claim(session, message)
            await self._offer(session)
            return None
        raise canonical.ContractError(f"{message.type!r} is not a worker message this service accepts")

    async def _reconcile(self, session: WorkerSession,
                         message: ReconcileResultMessage) -> None:
        try:
            self.store.record_reconciliation(
                message_id=message.message_id,
                job_id=message.job_id,
                sender=session.worker_id,
                outcome=message.outcome,
                result=message.result,
                result_sha256=message.result_sha256,
                reason_code=message.reason_code,
                detail=message.detail,
                now=self.now(),
            )
        except (JobNotFound, JobConflict):
            # The report is advisory: a job that is already gone or a conflicting replay is
            # dropped, never used to change state. The socket is not authority.
            pass
        record = self.store.find(message.job_id)
        if (
            session.active is not None
            and session.active.job_id == message.job_id
            and (record is None or record.state not in ACTIVE_STATES)
        ):
            session.finish(message.job_id)
            await self._offer(session)
        self._claim(session, message)
        return None

    # -- store and session helpers ---------------------------------------------------------------------

    def _transition(self, session: WorkerSession, message: JobMessage,
                    action: Callable[[], JobRecord]) -> JobRecord:
        try:
            current = self.store.get(message.job_id)
            if current.attempts != message.attempt:
                raise LeaseLost(
                    f"job {message.job_id!r} is on attempt {current.attempts}, "
                    f"not {message.attempt}"
                )
            return action()
        except LeaseLost as error:
            raise _Lost(self._lost(session, message, "lease_lost", error)) from error
        except JobConflict as error:
            raise _Lost(self._lost(session, message, "job_conflict", error)) from error
        except JobNotFound as error:
            raise _Lost(self._lost(session, message, "job_not_found", error)) from error

    def _lost(self, session: WorkerSession, message: JobMessage, reason_code: str,
              error: Exception) -> LeaseLostMessage:
        session.finish(message.job_id)
        return LeaseLostMessage.create(
            sender=self.sender,
            job_id=message.job_id,
            lease_token=message.lease_token,
            attempt=message.attempt,
            reason_code=reason_code,
            detail=str(error),
        )

    def _claim(self, session: WorkerSession, message: MessageType) -> None:
        # Apply first, then guard: every transition above is idempotent or fenced, so a crash
        # between the two only re-applies an effect. The reverse order could lose a report.
        self.store.claim_message(message.message_id, sender=session.worker_id, now=self.now())

    def _active_is_current(self, session: WorkerSession) -> bool:
        """A tracked job stops being current when its lease is gone, expired or re-owned."""
        if session.active is None:
            return False
        record = self.store.find(session.active.job_id)
        if (
            record is None
            or record.state not in ACTIVE_STATES
            or record.lease_token != session.active.lease_token
            or record.lease_expires_at is None
        ):
            return False
        return canonical.parse_iso(record.lease_expires_at) > canonical.utc(self.now())

    async def _offer(self, session: WorkerSession) -> LeaseMessage | None:
        if self._active_is_current(session):
            return None
        session.active = None
        lease = self.store.lease(
            worker_id=session.worker_id,
            lease_seconds=self.lease_seconds,
            kinds=session.kinds or None,
            now=self.now(),
        )
        if lease is None:
            return None
        message = LeaseMessage.create(
            sender=self.sender,
            job_id=lease.job_id,
            worker_id=session.worker_id,
            lease_token=lease.lease_token,
            expires_at=lease.expires_at,
            attempt=lease.attempt,
            kind=lease.kind,
            payload=lease.payload,
        )
        session.active = ActiveJob(
            job_id=lease.job_id,
            lease_token=lease.lease_token,
            attempt=lease.attempt,
            kind=lease.kind,
            expires_at=lease.expires_at,
        )
        try:
            await session.send(message_json(message))
        except Exception:
            # The row keeps its lease until expiry; we do not pretend the worker received it.
            session.active = None
            raise
        return message
