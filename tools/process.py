from typing import Any

from services.process_runner import run_process


def process_exec(
    executable: str,
    args: list[str] | None = None,
    cwd: str | None = None,
    timeout_seconds: int = 10,
) -> dict[str, Any]:
    """
    Execute a structured, policy-controlled native process.

    The actual security policy is enforced by
    services.process_runner.
    """

    return run_process(
        executable=executable,
        args=args,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )