from __future__ import annotations

import asyncio
import heapq
import itertools
import time

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

    Future HOT/WARM/COLD residency, measured eviction, and
    validated overlap can evolve behind this boundary.
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