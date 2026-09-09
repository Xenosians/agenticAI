from typing import Any

from services.process_runner import (
    run_process,
)


def workspace_mkdir(
    directory_name: str,
    cwd: str | None = None,
    timeout_seconds: int = 10,
) -> dict[str, Any]:
    """
    Create one direct-child directory through the governed
    native process runner.

    The caller supplies only the directory name.

    The trusted runner still owns:
    - mkdir executable selection
    - argument validation
    - workspace restriction
    - timeout policy
    - actual process execution
    """

    return run_process(
        executable="mkdir",
        args=[
            directory_name,
        ],
        cwd=cwd,
        timeout_seconds=(
            timeout_seconds
        ),
    )