from pathlib import Path

from learning.cli.crashlog import (
    run_with_crashlog,
)


def _logs(
    root: Path,
) -> list[Path]:

    return list(
        root.glob(
            "**/*.log"
        )
    )


def test_successful_command_removes_transient_log(
    tmp_path: Path,
):

    def callback():
        print(
            "successful output"
        )

        return 0

    code = (
        run_with_crashlog(
            "unit-success",
            callback,
            root=tmp_path,
            announce=False,
        )
    )

    assert code == 0

    assert (
        _logs(
            tmp_path
        )
        == []
    )


def test_exception_retains_traceback_and_output(
    tmp_path: Path,
):

    def callback():
        print(
            "output before crash"
        )

        raise RuntimeError(
            "synthetic gpu failure"
        )

    code = (
        run_with_crashlog(
            "unit-crash",
            callback,
            root=tmp_path,
            announce=False,
        )
    )

    assert code == 1

    logs = (
        _logs(
            tmp_path
        )
    )

    assert len(
        logs
    ) == 1

    content = (
        logs[
            0
        ]
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "output before crash"
        in content
    )

    assert (
        "RuntimeError"
        in content
    )

    assert (
        "synthetic gpu failure"
        in content
    )

    assert (
        "UNHANDLED EXCEPTION"
        in content
    )


def test_nonzero_exit_retains_log(
    tmp_path: Path,
):

    def callback():
        print(
            "controlled failure"
        )

        return 7

    code = (
        run_with_crashlog(
            "unit-nonzero",
            callback,
            root=tmp_path,
            announce=False,
        )
    )

    assert code == 7

    logs = (
        _logs(
            tmp_path
        )
    )

    assert len(
        logs
    ) == 1

    content = (
        logs[
            0
        ]
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "controlled failure"
        in content
    )

    assert (
        "exit_code=7"
        in content
    )


def test_tee_is_safe_after_log_file_closes(
    tmp_path: Path,
):

    import io

    from learning.cli.crashlog import (
        _TeeTextIO,
    )

    original = (
        io.StringIO()
    )

    log_path = (
        tmp_path
        / "closed.log"
    )

    log = log_path.open(
        "w",
        encoding="utf-8",
    )

    tee = (
        _TeeTextIO(
            original,
            log,
        )
    )

    tee.write(
        "before-close"
    )

    log.close()

    # Simulates an atexit library such as colorama retaining
    # the tee after run_with_crashlog has already closed its log.
    tee.write(
        "after-close"
    )

    tee.flush()

    assert (
        original.getvalue()
        == "before-closeafter-close"
    )

    assert (
        log_path.read_text(
            encoding="utf-8"
        )
        == "before-close"
    )
