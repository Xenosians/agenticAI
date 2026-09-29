from __future__ import annotations

import faulthandler
import json
import os
import platform
import re
import sys
import traceback

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path
from typing import (
    Callable,
    TextIO,
)

from learning.paths import (
    RUNTIME_LEARNING_ROOT,
)


DEFAULT_CRASHLOG_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "crashlogs"
)


_SECRET_ARGUMENT_PATTERN = re.compile(
    (
        r"(?:token|password|secret|api[-_]?key|"
        r"credential|authorization)"
    ),
    flags=re.IGNORECASE,
)


class _TeeTextIO:
    """
    Mirror Python stdout/stderr to both the original terminal
    stream and a persistent crash-log stream.
    """

    def __init__(
        self,
        original: TextIO,
        log: TextIO,
    ) -> None:

        self.original = original
        self.log = log

    def write(
        self,
        value: str,
    ) -> int:

        written = (
            self.original.write(
                value
            )
        )

        # Libraries such as colorama may retain a reference to the
        # stream until interpreter shutdown. At that point the
        # crash-log file may already be closed. The tee must degrade
        # back to the original terminal rather than raising from an
        # atexit callback.
        if not self.log.closed:

            try:

                self.log.write(
                    value
                )

                self.log.flush()

            except (
                OSError,
                ValueError,
            ):
                pass

        return written

    def flush(
        self,
    ) -> None:

        self.original.flush()

        if not self.log.closed:

            try:

                self.log.flush()

            except (
                OSError,
                ValueError,
            ):
                pass

    def __getattr__(
        self,
        name: str,
    ):
        return getattr(
            self.original,
            name,
        )


def _utc_now() -> str:

    return (
        datetime
        .now(
            timezone.utc
        )
        .isoformat()
    )


def _filename_timestamp() -> str:

    return (
        datetime
        .now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
    )


def _safe_name(
    value: str,
) -> str:

    normalized = re.sub(
        r"[^A-Za-z0-9._-]+",
        "-",
        value,
    ).strip(
        "-"
    )

    return (
        normalized
        or "command"
    )


def _redacted_argv() -> list[str]:

    result: list[str] = []

    redact_next = False

    for argument in sys.argv:

        if redact_next:

            result.append(
                "<redacted>"
            )

            redact_next = False

            continue

        if (
            argument.startswith(
                "--"
            )
            and "=" in argument
        ):

            key, _value = (
                argument.split(
                    "=",
                    1,
                )
            )

            if (
                _SECRET_ARGUMENT_PATTERN
                .search(
                    key
                )
            ):

                result.append(
                    key
                    + "=<redacted>"
                )

                continue

        result.append(
            argument
        )

        if (
            argument.startswith(
                "--"
            )
            and _SECRET_ARGUMENT_PATTERN
            .search(
                argument
            )
        ):

            redact_next = True

    return result


def _write_header(
    handle: TextIO,
    *,
    command_name: str,
) -> None:

    payload = {
        "command":
            command_name,

        "created_at":
            _utc_now(),

        "pid":
            os.getpid(),

        "cwd":
            os.getcwd(),

        "python":
            sys.version,

        "executable":
            sys.executable,

        "platform":
            platform.platform(),

        "argv":
            _redacted_argv(),
    }

    handle.write(
        "Agentic AI Crash Log\n"
    )

    handle.write(
        "====================\n"
    )

    handle.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    handle.write(
        "\n\n"
        "Runtime output\n"
        "--------------\n"
    )

    handle.flush()


def run_with_crashlog(
    command_name: str,
    callback: Callable[
        [],
        int | None,
    ],
    *,
    root: Path = (
        DEFAULT_CRASHLOG_ROOT
    ),
    announce: bool = True,
) -> int:
    """
    Execute one CLI with a failure-retained diagnostic log.

    Successful runs remove the temporary log.

    Failed runs retain:
      - stdout
      - stderr
      - Python traceback
      - faulthandler output for supported fatal interpreter faults

    Environment variables are deliberately NOT dumped because
    learning/runtime environments may contain credentials.
    """

    command_directory = (
        root
        .expanduser()
        .resolve()
        / _safe_name(
            command_name
        )
    )

    command_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        command_directory
        / (
            _filename_timestamp()
            + "-pid"
            + str(
                os.getpid()
            )
            + ".log"
        )
    )

    original_stdout = (
        sys.stdout
    )

    original_stderr = (
        sys.stderr
    )

    successful = False

    with log_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as handle:

        _write_header(
            handle,
            command_name=(
                command_name
            ),
        )

        sys.stdout = (
            _TeeTextIO(
                original_stdout,
                handle,
            )
        )

        sys.stderr = (
            _TeeTextIO(
                original_stderr,
                handle,
            )
        )

        try:

            faulthandler.enable(
                file=handle,
                all_threads=True,
            )

            if announce:

                print(
                    "[CRASHLOG] armed "
                    + str(
                        log_path
                    )
                    + " (retained only on failure)"
                )

            try:

                result = callback()

                exit_code = (
                    int(
                        result
                    )
                    if result
                    is not None
                    else 0
                )

            except KeyboardInterrupt:

                exit_code = 130

                print(
                    "\n[CRASHLOG] KeyboardInterrupt",
                    file=sys.stderr,
                )

            except SystemExit as exc:

                if exc.code is None:

                    exit_code = 0

                elif isinstance(
                    exc.code,
                    int,
                ):

                    exit_code = (
                        exc.code
                    )

                else:

                    exit_code = 1

                    print(
                        str(
                            exc.code
                        ),
                        file=sys.stderr,
                    )

            except BaseException:

                exit_code = 1

                print(
                    "\n"
                    "[CRASHLOG] UNHANDLED EXCEPTION",
                    file=sys.stderr,
                )

                traceback.print_exc(
                    file=sys.stderr,
                )

            if exit_code == 0:

                successful = True

            else:

                print(
                    "\n"
                    "[CRASHLOG] retained "
                    + str(
                        log_path
                    ),
                    file=sys.stderr,
                )

                print(
                    "[CRASHLOG] exit_code="
                    + str(
                        exit_code
                    ),
                    file=sys.stderr,
                )

        finally:

            try:

                if (
                    faulthandler
                    .is_enabled()
                ):

                    faulthandler.disable()

            except Exception:
                pass

            try:
                sys.stdout.flush()
            except Exception:
                pass

            try:
                sys.stderr.flush()
            except Exception:
                pass

            sys.stdout = (
                original_stdout
            )

            sys.stderr = (
                original_stderr
            )

    if successful:

        try:

            log_path.unlink(
                missing_ok=True
            )

        except Exception:

            # A failure to clean a successful transient log
            # must never convert a successful training/eval run
            # into a failed one.
            pass

    return exit_code
