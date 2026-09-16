import asyncio
import threading

from subagents.llm.runtime.scheduler import (
    GpuScheduler,
)


def test_scheduler_prioritizes_waiting_work():
    async def scenario():
        scheduler = (
            GpuScheduler()
        )

        first_started = (
            threading.Event()
        )

        release_first = (
            threading.Event()
        )

        order = []

        def first_work():
            first_started.set()

            release_first.wait(
                timeout=5
            )

            order.append(
                "first"
            )

        def low_priority_work():
            order.append(
                "low"
            )

        def high_priority_work():
            order.append(
                "high"
            )

        first_task = (
            asyncio.create_task(
                scheduler.run(
                    priority=50,
                    work=first_work,
                )
            )
        )

        started = (
            await asyncio.to_thread(
                first_started.wait,
                2,
            )
        )

        assert started is True

        # These two requests are now both waiting behind the
        # already-running first operation.
        low_task = (
            asyncio.create_task(
                scheduler.run(
                    priority=50,
                    work=(
                        low_priority_work
                    ),
                )
            )
        )

        high_task = (
            asyncio.create_task(
                scheduler.run(
                    priority=10,
                    work=(
                        high_priority_work
                    ),
                )
            )
        )

        # Give both tasks an opportunity to enter the queue.
        await asyncio.sleep(
            0
        )

        release_first.set()

        await asyncio.gather(
            first_task,
            low_task,
            high_task,
        )

        assert order == [
            "first",
            "high",
            "low",
        ]

    asyncio.run(
        scenario()
    )


def test_scheduler_does_not_overlap_after_cancellation():
    async def scenario():
        scheduler = (
            GpuScheduler()
        )

        first_started = (
            threading.Event()
        )

        release_first = (
            threading.Event()
        )

        second_started = (
            threading.Event()
        )

        def first_work():
            first_started.set()

            release_first.wait(
                timeout=5
            )

            return "first"

        def second_work():
            second_started.set()

            return "second"

        first_task = (
            asyncio.create_task(
                scheduler.run(
                    priority=10,
                    work=first_work,
                )
            )
        )

        started = (
            await asyncio.to_thread(
                first_started.wait,
                2,
            )
        )

        assert started is True

        second_task = (
            asyncio.create_task(
                scheduler.run(
                    priority=20,
                    work=second_work,
                )
            )
        )

        await asyncio.sleep(
            0
        )

        first_task.cancel()

        # Cancellation must NOT make the GPU slot available while
        # first_work() is still running in its thread.
        await asyncio.sleep(
            0.05
        )

        assert (
            second_started.is_set()
            is False
        )

        assert (
            scheduler.active
            is True
        )

        release_first.set()

        try:
            await first_task

        except asyncio.CancelledError:
            pass

        second_result = (
            await second_task
        )

        assert (
            second_result
            == "second"
        )

        assert (
            second_started.is_set()
            is True
        )

        assert (
            scheduler.active
            is False
        )

    asyncio.run(
        scenario()
    )