from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar

from dataclasses import (
    dataclass,
    field,
)

from enum import IntEnum

from typing import (
    Callable,
    Generic,
    TypeVar,
)


T = TypeVar(
    "T"
)


class InferencePriority(
    IntEnum
):
    """
    Lower number means higher scheduling priority.

    Scheduling is deliberately non-preemptive.

    A generation that has already entered the GPU boundary is
    allowed to finish. Priority only reorders work that is
    waiting for admission.
    """

    STARTUP = 0

    SPECIALIST = 10

    HUB_ROUTING = 20

    PRIMARY_RESPONSE = 30


@dataclass(
    order=True
)
class _GpuWaiter:
    priority: int

    sequence: int

    future: asyncio.Future = field(
        compare=False
    )

    granted: bool = field(
        default=False,
        compare=False,
    )


@dataclass(
    frozen=True
)
class SchedulerRunMetrics:
    queue_wait_seconds: float

    execution_seconds: float

    total_seconds: float


@dataclass(
    frozen=True
)
class ScheduledResult(
    Generic[T]
):
    value: T

    metrics: SchedulerRunMetrics


class _JobTurn:
    """One job's place in the line for the local model."""

    def __init__(self) -> None:
        self.line: LocalModelTurns | None = None
        self.granted: asyncio.Future | None = None
        self.holds_model = False


# The job whose steps run in this context; None outside a job (startup
# warm-up, approved actions), whose steps never wait for a turn.
_current_job_turn: ContextVar[_JobTurn | None] = ContextVar("current_job_turn", default=None)


@contextmanager
def job_turn():
    """Run one job's steps; its local-model steps take turns with other jobs.

    A job keeps the local model from its first local step until it ends, so
    two local jobs do not interleave their steps on one GPU and both finish
    late. Steps on an outbound connection never wait for this turn.
    """
    turn = _JobTurn()
    token = _current_job_turn.set(turn)
    try:
        yield
    finally:
        _current_job_turn.reset(token)
        if turn.line is not None:
            turn.line.leave(turn)


class LocalModelTurns:
    """First-come line of jobs for the local model; one job holds it at a time."""

    def __init__(self) -> None:
        self._holder: _JobTurn | None = None
        self._line: deque[_JobTurn] = deque()

    async def enter(self, on_wait: Callable[[], None] | None = None) -> bool:
        """Wait until the current job holds the local model; True if it had to wait."""
        turn = _current_job_turn.get()
        if turn is None or turn.holds_model:
            return False
        turn.line = self
        if self._holder is None:
            self._holder, turn.holds_model = turn, True
            return False
        if turn.granted is None:
            turn.granted = asyncio.get_running_loop().create_future()
            self._line.append(turn)
        if on_wait is not None:
            on_wait()
        # Parallel steps of one job wait on the same grant and share the turn.
        await asyncio.shield(turn.granted)
        return True

    def leave(self, turn: _JobTurn) -> None:
        if turn in self._line:
            self._line.remove(turn)
        if turn.granted is not None and not turn.granted.done():
            turn.granted.cancel()
        if self._holder is turn:
            self._holder = None
            if self._line:
                following = self._line.popleft()
                self._holder, following.holds_model = following, True
                if following.granted is not None:
                    following.granted.set_result(None)


class GpuScheduler:
    """
    Process-local serialized inference admission boundary.

    Current policy:

    - exactly one admitted model operation at a time
    - queued requests are ordered by priority
    - equal-priority requests remain FIFO
    - already-running inference is never preempted
    - blocking model work runs outside the asyncio event loop
    - cancellation never releases the GPU slot while blocking
      model work is still executing

    Observability is intentionally measured here because this is
    the resource-admission boundary.

    Model residency / eviction is deliberately owned by
    ModelManager rather than this scheduler. Future validated
    overlap can evolve behind this admission boundary without
    merging the two responsibilities.
    """

    def __init__(
        self,
    ) -> None:
        self._lock = (
            asyncio.Lock()
        )

        self._waiters: list[
            _GpuWaiter
        ] = []

        self._sequence = (
            itertools.count()
        )

        self._active = False

        # Jobs take turns on the local model; see job_turn().
        self.job_turns = LocalModelTurns()

    @property
    def active(
        self,
    ) -> bool:
        return self._active

    @property
    def queue_depth(
        self,
    ) -> int:
        return sum(
            1

            for waiter
            in self._waiters

            if not (
                waiter.future
                .cancelled()
            )
        )

    async def run(
        self,
        *,
        priority: int,
        work: Callable[
            [],
            T,
        ],
    ) -> T:
        """
        Compatibility entry point.

        Existing callers receive the original value while
        observability-aware callers may use run_observed().
        """

        result = (
            await self.run_observed(
                priority=(
                    priority
                ),
                work=(
                    work
                ),
            )
        )

        return result.value

    async def run_observed(
        self,
        *,
        priority: int,
        work: Callable[
            [],
            T,
        ],
    ) -> ScheduledResult[T]:
        submitted_at = (
            time.perf_counter()
        )

        await self._acquire(
            priority
        )

        admitted_at = (
            time.perf_counter()
        )

        worker_task: (
            asyncio.Task | None
        ) = None

        try:
            worker_task = (
                asyncio.create_task(
                    asyncio.to_thread(
                        work
                    )
                )
            )

            try:
                value = (
                    await asyncio.shield(
                        worker_task
                    )
                )

            except asyncio.CancelledError:
                try:
                    await worker_task

                except Exception:
                    pass

                raise

            completed_at = (
                time.perf_counter()
            )

            metrics = (
                SchedulerRunMetrics(
                    queue_wait_seconds=(
                        admitted_at
                        - submitted_at
                    ),

                    execution_seconds=(
                        completed_at
                        - admitted_at
                    ),

                    total_seconds=(
                        completed_at
                        - submitted_at
                    ),
                )
            )

            return (
                ScheduledResult(
                    value=value,
                    metrics=metrics,
                )
            )

        finally:
            await self._release()

    async def _acquire(
        self,
        priority: int,
    ) -> None:
        loop = (
            asyncio
            .get_running_loop()
        )

        future = (
            loop.create_future()
        )

        waiter = _GpuWaiter(
            priority=int(
                priority
            ),

            sequence=next(
                self._sequence
            ),

            future=future,
        )

        async with self._lock:
            heapq.heappush(
                self._waiters,
                waiter,
            )

            self._grant_next_locked()

        try:
            await future

        except asyncio.CancelledError:
            async with self._lock:
                if waiter.granted:
                    self._active = False

                if not future.done():
                    future.cancel()

                self._grant_next_locked()

            raise

    async def _release(
        self,
    ) -> None:
        async with self._lock:
            self._active = False

            self._grant_next_locked()

    def _grant_next_locked(
        self,
    ) -> None:
        if self._active:
            return

        while self._waiters:
            waiter = (
                heapq.heappop(
                    self._waiters
                )
            )

            if (
                waiter.future
                .cancelled()
            ):
                continue

            self._active = True

            waiter.granted = True

            waiter.future.set_result(
                None
            )

            return