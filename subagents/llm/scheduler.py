from __future__ import annotations

import asyncio
import heapq
import itertools

from dataclasses import (
    dataclass,
    field,
)

from enum import IntEnum

from typing import (
    Callable,
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

    Future VRAM measurement, HOT/WARM/COLD residency, measured
    eviction, and validated overlap can evolve behind this
    boundary.
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
        await self._acquire(
            priority
        )

        worker_task: (
            asyncio.Task | None
        ) = None

        try:
            # ----------------------------------------------------
            # Blocking HF/PyTorch inference must not block the
            # FastAPI event loop.
            #
            # Keep an explicit Task around the thread operation.
            # asyncio.to_thread() itself cannot stop the native
            # work when its awaiting coroutine is cancelled.
            # ----------------------------------------------------

            worker_task = (
                asyncio.create_task(
                    asyncio.to_thread(
                        work
                    )
                )
            )

            try:
                return await asyncio.shield(
                    worker_task
                )

            except asyncio.CancelledError:
                # ------------------------------------------------
                # Critical GPU safety rule:
                #
                # The Python coroutine may be cancelled, but the
                # underlying model call is still running in its
                # worker thread.
                #
                # Do not release GPU admission until that native
                # operation has actually finished.
                # ------------------------------------------------

                try:
                    await worker_task

                except Exception:
                    # The caller was cancelled, so cancellation
                    # remains the externally visible outcome.
                    #
                    # The worker exception is intentionally not
                    # substituted for CancelledError here.
                    pass

                raise

        finally:
            # ----------------------------------------------------
            # At this point the admitted blocking operation is no
            # longer running.
            # ----------------------------------------------------

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
                # If admission was granted but the coroutine was
                # cancelled before it started its blocking work,
                # return the slot immediately.
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