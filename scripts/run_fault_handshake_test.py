from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


# ============================================================
# Make repository root importable
# ============================================================

AI_ROOT = Path(__file__).resolve().parents[1]

if str(AI_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(AI_ROOT),
    )


import httpx

from agent.completion_outbox import (
    CompletionOutbox,
)


# ============================================================
# Paths
# ============================================================


BACKEND_ROOT = Path(
    os.getenv(
        "ITSM_BACKEND_ROOT",
        "/mnt/c/project/agenticBackend/itsm_backend",
    )
)

OUTBOX_PATH = (
    AI_ROOT
    / ".runtime"
    / "completion_outbox.sqlite3"
)


# ============================================================
# Services
# ============================================================


PHOENIX_URL = "http://127.0.0.1:4000"
PROXY_URL = "http://127.0.0.1:8000"
REAL_AI_URL = "http://127.0.0.1:8003"

TEST_USER = "jdoe"
TEST_MESSAGE = "Is jdoe locked?"

FAIL_COMPLETIONS = "3"

STARTUP_TIMEOUT = 240.0
JOB_TIMEOUT = 240.0
POLL_SECONDS = 1.0


# ============================================================
# Output
# ============================================================


def banner(text: str) -> None:
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def info(text: str) -> None:
    print(f"[INFO] {text}")


def passed(text: str) -> None:
    print(f"[PASS] {text}")


def failed(text: str) -> None:
    print(f"[FAIL] {text}")


# ============================================================
# Managed process
# ============================================================


class ManagedProcess:
    def __init__(
        self,
        name: str,
        command: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.cwd = cwd
        self.env = env
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        info(
            f"Starting {self.name}: "
            f"{' '.join(self.command)}"
        )

        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

    def alive(self) -> bool:
        return (
            self.process is not None
            and self.process.poll() is None
        )

    def terminate(self) -> None:
        if not self.alive():
            return

        assert self.process is not None

        info(
            f"Stopping {self.name}"
        )

        try:
            os.killpg(
                os.getpgid(
                    self.process.pid
                ),
                signal.SIGTERM,
            )

            self.process.wait(
                timeout=10
            )

        except subprocess.TimeoutExpired:
            os.killpg(
                os.getpgid(
                    self.process.pid
                ),
                signal.SIGKILL,
            )

            self.process.wait(
                timeout=5
            )

    def read_available(self) -> list[str]:
        if (
            self.process is None
            or self.process.stdout is None
        ):
            return []

        lines: list[str] = []

        while True:
            line = self.process.stdout.readline()

            if not line:
                break

            lines.append(
                line.rstrip()
            )

            # Avoid blocking indefinitely after one
            # available line.
            if (
                self.process.poll()
                is not None
            ):
                continue

            break

        return lines


# ============================================================
# HTTP helpers
# ============================================================


async def wait_for_http(
    name: str,
    url: str,
    *,
    expected_status: int = 200,
    timeout: float = STARTUP_TIMEOUT,
) -> None:
    deadline = (
        time.monotonic()
        + timeout
    )

    async with httpx.AsyncClient(
        timeout=5.0,
    ) as client:
        while (
            time.monotonic()
            < deadline
        ):
            try:
                response = await client.get(
                    url
                )

                if (
                    response.status_code
                    == expected_status
                ):
                    passed(
                        f"{name} ready"
                    )

                    return

            except httpx.HTTPError:
                pass

            await asyncio.sleep(1)

    raise TimeoutError(
        f"{name} failed to become ready"
    )


async def create_job(
    client: httpx.AsyncClient,
) -> str:
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
            "Job creation failed: "
            f"{response.status_code} "
            f"{response.text}"
        )

    body = response.json()

    job_id = body["job_id"]

    passed(
        f"Created real job {job_id}"
    )

    return job_id


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
            f"{PROXY_URL}"
            f"/__faults/jobs/"
            f"{job_id}"
        )
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# Outbox
# ============================================================


def outbox() -> CompletionOutbox:
    box = CompletionOutbox(
        OUTBOX_PATH
    )

    box.initialize()

    return box


def outbox_contains(
    job_id: str,
) -> bool:
    return any(
        entry.job_id == job_id
        for entry
        in outbox().list_pending()
    )


# ============================================================
# Validation
# ============================================================


async def run_fault_scenario() -> None:
    banner(
        "RUNNING TWO-HANDSHAKE FAULT TEST"
    )

    async with httpx.AsyncClient(
        timeout=10.0,
    ) as client:
        job_id = await create_job(
            client
        )

        deadline = (
            time.monotonic()
            + JOB_TIMEOUT
        )

        saw_execution = False
        saw_fault = False
        saw_outbox = False
        saw_forward = False

        last_status = None
        last_failure_count = 0

        while (
            time.monotonic()
            < deadline
        ):
            job = await get_job(
                client,
                job_id,
            )

            fault = await get_fault_state(
                client,
                job_id,
            )

            status = job.get(
                "status"
            )

            execution_requests = (
                fault.get(
                    "execution_requests",
                    0,
                )
            )

            failure_count = (
                fault.get(
                    "completion_failures",
                    0,
                )
            )

            forward_count = (
                fault.get(
                    "completion_forwards",
                    0,
                )
            )

            if status != last_status:
                info(
                    f"Phoenix state -> {status}"
                )

                last_status = status

            if execution_requests > 0:
                saw_execution = True

            if (
                failure_count
                > last_failure_count
            ):
                info(
                    "Injected callback failure "
                    f"#{failure_count}"
                )

                last_failure_count = (
                    failure_count
                )

            if failure_count > 0:
                saw_fault = True

            if (
                outbox_contains(job_id)
                and not saw_outbox
            ):
                saw_outbox = True

                passed(
                    "Completion observed "
                    "persisted in SQLite outbox"
                )

            if forward_count > 0:
                saw_forward = True

            if status == "completed":
                break

            if status == "failed":
                raise RuntimeError(
                    "Job entered failed state"
                )

            await asyncio.sleep(
                POLL_SECONDS
            )

        else:
            raise TimeoutError(
                "Timed out waiting for "
                f"job {job_id}"
            )

        banner(
            "VALIDATING RESULTS"
        )

        if not saw_execution:
            raise AssertionError(
                "Proxy never observed "
                "Phoenix -> Python execution"
            )

        passed(
            "Phoenix -> Python handshake observed"
        )

        if not saw_fault:
            raise AssertionError(
                "No completion fault was injected"
            )

        passed(
            "Python -> Phoenix callback "
            "was intentionally faulted"
        )

        if not saw_outbox:
            raise AssertionError(
                "Completion was never seen "
                "inside durable outbox"
            )

        passed(
            "Completion remained durable "
            "during injected outage"
        )

        if not saw_forward:
            raise AssertionError(
                "Completion was never "
                "forwarded after recovery"
            )

        passed(
            "Completion eventually reached Phoenix"
        )

        final_job = await get_job(
            client,
            job_id,
        )

        if (
            final_job.get("status")
            != "completed"
        ):
            raise AssertionError(
                "Final durable job state "
                "is not completed"
            )

        passed(
            "Surreal-backed job is completed"
        )

        if outbox_contains(
            job_id
        ):
            raise AssertionError(
                "Outbox entry still exists "
                "after Phoenix ACK"
            )

        passed(
            "Outbox entry removed "
            "after Phoenix ACK"
        )

        if (
            final_job.get("attempts")
            != 1
        ):
            raise AssertionError(
                "Unexpected job attempt count: "
                f"{final_job.get('attempts')}"
            )

        passed(
            "Job stayed on attempt 1"
        )

        print()
        print("Fault timeline:")
        print("-" * 72)

        for event in fault.get(
            "events",
            [],
        ):
            print(event)

        print()
        passed(
            f"job_id={job_id}"
        )


# ============================================================
# Main test lifecycle
# ============================================================


async def run_suite() -> None:
    if not BACKEND_ROOT.exists():
        raise RuntimeError(
            "Backend directory does not exist: "
            f"{BACKEND_ROOT}"
        )

    python_executable = (
        sys.executable
    )

    base_env = os.environ.copy()

    # --------------------------------------------------------
    # Real AI
    #
    # Its Phoenix callback points to the fault proxy.
    # --------------------------------------------------------

    ai_env = base_env.copy()

    ai_env[
        "PHOENIX_BASE_URL"
    ] = PROXY_URL

    real_ai = ManagedProcess(
        name="real AI",
        command=[
            python_executable,
            "-m",
            "uvicorn",
            "api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8003",
        ],
        cwd=AI_ROOT,
        env=ai_env,
    )

    # --------------------------------------------------------
    # Fault proxy
    # --------------------------------------------------------

    proxy_env = base_env.copy()

    proxy_env[
        "REAL_AI_URL"
    ] = REAL_AI_URL

    proxy_env[
        "REAL_PHOENIX_URL"
    ] = PHOENIX_URL

    proxy_env[
        "FAIL_COMPLETIONS"
    ] = FAIL_COMPLETIONS

    proxy = ManagedProcess(
        name="fault proxy",
        command=[
            python_executable,
            "-m",
            "uvicorn",
            "scripts.fault_proxy:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=AI_ROOT,
        env=proxy_env,
    )

    # --------------------------------------------------------
    # Phoenix
    # --------------------------------------------------------

    phoenix = ManagedProcess(
        name="Phoenix",
        command=[
            "mix",
            "phx.server",
        ],
        cwd=BACKEND_ROOT,
        env=base_env.copy(),
    )

    processes = [
        real_ai,
        proxy,
        phoenix,
    ]

    try:
        banner(
            "STARTING TEST ENVIRONMENT"
        )

        # AI loads slowly, so start it first.
        real_ai.start()

        await wait_for_http(
            "real AI",
            f"{REAL_AI_URL}/ready",
        )

        proxy.start()

        await wait_for_http(
            "fault proxy",
            f"{PROXY_URL}/ready",
        )

        phoenix.start()

        await wait_for_http(
            "Phoenix",
            f"{PHOENIX_URL}/api/health",
        )

        banner(
            "ENVIRONMENT READY"
        )

        await run_fault_scenario()

        banner(
            "HANDSHAKE FAULT TEST PASSED"
        )

    finally:
        banner(
            "CLEANING UP TEST ENVIRONMENT"
        )

        for process in reversed(
            processes
        ):
            process.terminate()


def main() -> None:
    try:
        asyncio.run(
            run_suite()
        )

    except KeyboardInterrupt:
        print()
        failed(
            "Test interrupted"
        )

        sys.exit(130)

    except Exception as exc:
        print()
        failed(
            repr(exc)
        )

        banner(
            "HANDSHAKE FAULT TEST FAILED"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()