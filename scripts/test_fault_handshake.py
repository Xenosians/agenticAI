from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

import httpx

from agent.completion_outbox import (
    CompletionOutbox,
)


PHOENIX_URL = (
    "http://127.0.0.1:4000"
)

FAULT_PROXY_URL = (
    "http://127.0.0.1:8000"
)

OUTBOX_PATH = (
    ".runtime/completion_outbox.sqlite3"
)

TEST_MESSAGE = (
    "Is jdoe locked?"
)

TEST_USER = "jdoe"

TIMEOUT_SECONDS = 180.0
POLL_SECONDS = 1.0


# ============================================================
# Output
# ============================================================


def stage(
    text: str,
) -> None:
    print()
    print(
        "=" * 68
    )
    print(
        f"[TEST] {text}"
    )
    print(
        "=" * 68
    )


def ok(
    text: str,
) -> None:
    print(
        f"[PASS] {text}"
    )


def info(
    text: str,
) -> None:
    print(
        f"[INFO] {text}"
    )


def fail(
    text: str,
) -> None:
    print(
        f"[FAIL] {text}"
    )


# ============================================================
# Helpers
# ============================================================


def outbox_contains(
    outbox: CompletionOutbox,
    job_id: str,
) -> bool:
    return any(
        entry.job_id == job_id
        for entry
        in outbox.list_pending()
    )


async def get_job(
    client: httpx.AsyncClient,
    job_id: str,
) -> dict[str, Any]:
    response = await client.get(
        (
            f"{PHOENIX_URL}"
            f"/api/v1/jobs/"
            f"{job_id}"
        )
    )

    response.raise_for_status()

    return response.json()


async def get_fault_state(
    client: httpx.AsyncClient,
    job_id: str,
) -> dict[str, Any]:
    response = await client.get(
        (
            f"{FAULT_PROXY_URL}"
            f"/__faults/jobs/"
            f"{job_id}"
        )
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# Preflight
# ============================================================


async def preflight(
    client: httpx.AsyncClient,
) -> None:
    stage(
        "Preflight"
    )

    response = await client.get(
        f"{PHOENIX_URL}/api/health"
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Phoenix is not healthy"
        )

    ok(
        "Phoenix reachable"
    )

    response = await client.get(
        f"{FAULT_PROXY_URL}/ready"
    )

    if response.status_code != 200:
        raise RuntimeError(
            "AI through fault proxy "
            "is not ready"
        )

    ok(
        "Fault proxy -> real AI ready"
    )

    response = await client.get(
        f"{FAULT_PROXY_URL}/__faults"
    )

    response.raise_for_status()

    config = response.json()

    info(
        "Fault configuration: "
        f"{config}"
    )


# ============================================================
# Create real Phoenix job
# ============================================================


async def create_job(
    client: httpx.AsyncClient,
) -> str:
    stage(
        "Creating real Phoenix job"
    )

    response = await client.post(
        f"{PHOENIX_URL}/api/v1/jobs",
        json={
            "user_id":
                TEST_USER,
            "message":
                TEST_MESSAGE,
        },
    )

    if response.status_code != 202:
        raise RuntimeError(
            "Phoenix job creation "
            f"failed: {response.status_code} "
            f"{response.text}"
        )

    body = response.json()

    job_id = body[
        "job_id"
    ]

    ok(
        f"Created job_id={job_id}"
    )

    return job_id


# ============================================================
# Test
# ============================================================


async def run_test() -> None:
    outbox = CompletionOutbox(
        OUTBOX_PATH
    )

    outbox.initialize()

    async with httpx.AsyncClient(
        timeout=15.0,
    ) as client:
        await preflight(
            client
        )

        job_id = await create_job(
            client
        )

        stage(
            "Waiting for Phoenix -> Python"
        )

        deadline = (
            time.monotonic()
            + TIMEOUT_SECONDS
        )

        saw_processing = False
        saw_fault = False
        saw_durable_outbox = False
        saw_completion_forward = False

        previous_status = None
        previous_failures = -1

        while (
            time.monotonic()
            < deadline
        ):
            job = await get_job(
                client,
                job_id,
            )

            fault = (
                await get_fault_state(
                    client,
                    job_id,
                )
            )

            status = job.get(
                "status"
            )

            failures = fault.get(
                "completion_failures",
                0,
            )

            forwards = fault.get(
                "completion_forwards",
                0,
            )

            executions = fault.get(
                "execution_requests",
                0,
            )

            in_outbox = outbox_contains(
                outbox,
                job_id,
            )

            if status != previous_status:
                info(
                    "Phoenix state: "
                    f"{status}"
                )

                previous_status = (
                    status
                )

            if executions > 0:
                saw_processing = True

            if failures != previous_failures:
                if failures > 0:
                    info(
                        "Injected completion "
                        f"failure #{failures}"
                    )

                previous_failures = (
                    failures
                )

            if failures > 0:
                saw_fault = True

            if (
                in_outbox
                and not saw_durable_outbox
            ):
                saw_durable_outbox = True

                ok(
                    "Completion survived in "
                    "SQLite outbox while "
                    "Phoenix callback was blocked"
                )

            if forwards > 0:
                saw_completion_forward = (
                    True
                )

            if status == "completed":
                stage(
                    "Final verification"
                )

                if not saw_processing:
                    raise RuntimeError(
                        "Proxy never observed "
                        "Phoenix -> Python execution"
                    )

                ok(
                    "Proxy observed real "
                    "Phoenix -> Python execution"
                )

                if not saw_fault:
                    raise RuntimeError(
                        "No completion fault "
                        "was injected"
                    )

                ok(
                    "Completion callback fault "
                    "was injected"
                )

                if not saw_durable_outbox:
                    raise RuntimeError(
                        "Completion was never "
                        "observed in durable outbox"
                    )

                ok(
                    "Durable outbox held "
                    "the completion"
                )

                if not saw_completion_forward:
                    raise RuntimeError(
                        "Completion was never "
                        "forwarded after fault"
                    )

                ok(
                    "Completion eventually "
                    "forwarded to Phoenix"
                )

                if outbox_contains(
                    outbox,
                    job_id,
                ):
                    raise RuntimeError(
                        "Outbox entry still "
                        "exists after Phoenix ACK"
                    )

                ok(
                    "Outbox entry removed "
                    "after Phoenix ACK"
                )

                if job.get(
                    "attempts"
                ) != 1:
                    raise RuntimeError(
                        "Unexpected job attempt: "
                        f"{job.get('attempts')}"
                    )

                ok(
                    "Job completed on attempt 1"
                )

                print()
                print(
                    "FAULT TIMELINE"
                )
                print(
                    "-" * 68
                )

                for event in fault.get(
                    "events",
                    [],
                ):
                    print(
                        event
                    )

                print()
                print(
                    "=" * 68
                )
                print(
                    "HANDSHAKE FAULT TEST PASSED"
                )
                print(
                    "=" * 68
                )

                return

            if status == "failed":
                raise RuntimeError(
                    "Job entered failed state"
                )

            await asyncio.sleep(
                POLL_SECONDS
            )

        raise TimeoutError(
            f"Timed out waiting for "
            f"job_id={job_id}"
        )


# ============================================================
# Main
# ============================================================


def main() -> None:
    try:
        asyncio.run(
            run_test()
        )

    except Exception as exc:
        print()
        fail(
            repr(exc)
        )

        print()
        print(
            "=" * 68
        )
        print(
            "HANDSHAKE FAULT TEST FAILED"
        )
        print(
            "=" * 68
        )

        sys.exit(1)


if __name__ == "__main__":
    main()