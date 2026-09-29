from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from learning.continual.storage import (
    immutable_write_json,
    load_jsonl_models,
    sha256_file,
)

from learning.evaluation.developer_behavioral_holdout import (
    DeveloperBehavioralHoldoutManifest,
    DeveloperBehavioralHoldoutRecord,
)

from learning.paths import (
    EVALUATIONS_ROOT,
)


DEFAULT_DEVELOPER_BEHAVIORAL_SANDBOX_ROOT = (
    EVALUATIONS_ROOT
    / "developer-behavioral-sandbox-runs"
)


MAX_PATCH_BYTES = (
    2
    * 1024
    * 1024
)

MAX_TEST_PATCH_BYTES = (
    8
    * 1024
    * 1024
)

MAX_CAPTURE_CHARS = (
    1_000_000
)

MAX_TIMEOUT_SECONDS = 3600

MAX_MEMORY_MB = 16384

MAX_CPUS = 8.0


_REPOSITORY_PATTERN = re.compile(
    r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$"
)


class DockerImageIdentity(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    requested: str

    image_id: str

    repo_digests: list[
        str
    ] = Field(
        default_factory=list
    )


class DeveloperBehavioralSandboxReport(
    BaseModel
):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default=(
            "developer-behavioral-sandbox-run.v1"
        ),
        alias="schema",
    )

    run_id: str

    created_at: str

    behavioral_holdout_id: str

    behavioral_records_sha256: str

    record_id: str

    source_record_id: (
        str
        | None
    ) = None

    source_identity: str

    repository: str

    base_commit: str

    image: DockerImageIdentity

    candidate_patch_sha256: str

    test_patch_sha256: str

    test_command_count: int

    sandbox_network: str = "none"

    cap_drop_all: bool = True

    no_new_privileges: bool = True

    patch_mount_read_only: bool = True

    pids_limit: int

    memory_mb: int

    cpus: float

    timeout_seconds: int

    duration_ms: int = Field(
        ge=0
    )

    container_exit_code: (
        int
        | None
    ) = None

    timed_out: bool = False

    output_truncated: bool = False

    last_stage: (
        str
        | None
    ) = None

    candidate_patch_applied: bool = False

    test_patch_applied: bool = False

    test_execution_started: bool = False

    test_process_exit_success: bool = False

    stdout_sha256: str

    stderr_sha256: str

    stdout_path: str

    stderr_path: str

    # Parsing FAIL_TO_PASS / PASS_TO_PASS is the next gate.
    behavioral_scoring_complete: bool = False

    promotion_authorized: bool = False

    output_directory: str


class DeveloperBehavioralSandboxResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    report: DeveloperBehavioralSandboxReport

    report_path: str

    stdout_path: str

    stderr_path: str


def _utc_now() -> str:

    return (
        datetime
        .now(
            timezone.utc
        )
        .isoformat()
    )


def _sha256_text(
    value: str,
) -> str:

    return (
        hashlib
        .sha256(
            value.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


def _require_docker() -> str:

    executable = (
        shutil.which(
            "docker"
        )
    )

    if executable is None:

        raise RuntimeError(
            "Docker is required for developer "
            "behavioral evaluation."
        )

    return executable


def _load_behavioral_holdout(
    directory: Path,
) -> tuple[
    DeveloperBehavioralHoldoutManifest,
    list[
        DeveloperBehavioralHoldoutRecord
    ],
]:

    directory = (
        directory
        .expanduser()
        .resolve()
    )

    manifest_path = (
        directory
        / "manifest.json"
    )

    records_path = (
        directory
        / "records.jsonl"
    )

    if not (
        manifest_path.is_file()
        and records_path.is_file()
    ):

        raise ValueError(
            "Behavioral holdout artifact is incomplete."
        )

    manifest = (
        DeveloperBehavioralHoldoutManifest
        .model_validate_json(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    )

    if (
        manifest.evaluation_only
        is not True

        or manifest.training_eligible
        is not False

        or manifest.training_authorized
        is not False

        or manifest.promotion_authorized
        is not False

        or manifest.gold_patch_included
        is not False

        or manifest.test_patch_included
        is not True
    ):

        raise PermissionError(
            "Behavioral holdout governance "
            "flags are invalid."
        )

    if (
        sha256_file(
            records_path
        )
        != manifest.records_sha256
    ):

        raise ValueError(
            "Behavioral holdout records "
            "SHA-256 verification failed."
        )

    records = (
        load_jsonl_models(
            records_path,
            DeveloperBehavioralHoldoutRecord,
        )
    )

    if (
        len(
            records
        )
        != manifest.holdout_count
    ):

        raise ValueError(
            "Behavioral holdout record count "
            "does not match its manifest."
        )

    return (
        manifest,
        records,
    )


def _find_case(
    records: list[
        DeveloperBehavioralHoldoutRecord
    ],
    case_id: str,
) -> DeveloperBehavioralHoldoutRecord:

    case_id = (
        case_id.strip()
    )

    for record in records:

        if case_id in {
            record.record_id,
            record.source_record_id,
            record.source_identity,
        }:

            return record

    raise ValueError(
        "Unknown behavioral holdout case: "
        + case_id
    )


def behavioral_case_summaries(
    directory: Path,
) -> list[
    dict[str, Any]
]:

    _manifest, records = (
        _load_behavioral_holdout(
            directory
        )
    )

    result = []

    for record in records:

        install_config = (
            record.install_config
        )

        test_cmd = (
            install_config.get(
                "test_cmd"
            )
        )

        if isinstance(
            test_cmd,
            str,
        ):

            command_count = 1

        elif isinstance(
            test_cmd,
            list,
        ):

            command_count = len(
                [
                    item
                    for item
                    in test_cmd
                    if (
                        isinstance(
                            item,
                            str,
                        )
                        and item.strip()
                    )
                ]
            )

        else:

            command_count = 0

        result.append(
            {
                "case_id":
                    (
                        record.source_record_id
                        or record.record_id
                    ),

                "language":
                    record.language,

                "repository":
                    record.repository,

                "image":
                    record.image_name,

                "log_parser":
                    install_config.get(
                        "log_parser"
                    ),

                "test_command_count":
                    command_count,

                "fail_to_pass":
                    len(
                        record.fail_to_pass
                    ),

                "pass_to_pass":
                    len(
                        record.pass_to_pass
                    ),
            }
        )

    return result


def _test_commands(
    record: DeveloperBehavioralHoldoutRecord,
) -> list[str]:

    value = (
        record
        .install_config
        .get(
            "test_cmd"
        )
    )

    if isinstance(
        value,
        str,
    ):

        value = [
            value
        ]

    if not isinstance(
        value,
        list,
    ):

        raise ValueError(
            "Behavioral case has no valid "
            "install_config.test_cmd."
        )

    commands = [
        item.strip()

        for item
        in value

        if (
            isinstance(
                item,
                str,
            )
            and item.strip()
        )
    ]

    if not commands:

        raise ValueError(
            "Behavioral case has an empty "
            "install_config.test_cmd."
        )

    return commands


def _read_candidate_patch(
    path: Path,
) -> str:

    path = (
        path
        .expanduser()
        .resolve()
    )

    if not path.is_file():

        raise ValueError(
            "Candidate patch does not exist: "
            + str(
                path
            )
        )

    size = (
        path.stat()
        .st_size
    )

    if (
        size <= 0
        or size > MAX_PATCH_BYTES
    ):

        raise ValueError(
            "Candidate patch size must be between "
            f"1 and {MAX_PATCH_BYTES} bytes."
        )

    patch = (
        path.read_text(
            encoding="utf-8"
        )
    )

    if not patch.strip():

        raise ValueError(
            "Candidate patch is empty."
        )

    return patch


def _bounded_subprocess(
    command: list[str],
    *,
    timeout_seconds: int,
) -> tuple[
    int | None,
    str,
    str,
    bool,
    bool,
]:

    process = (
        subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
    )

    stdout_parts: list[
        str
    ] = []

    stderr_parts: list[
        str
    ] = []

    stdout_count = 0
    stderr_count = 0

    truncated = False

    lock = (
        threading.Lock()
    )

    def drain(
        stream,
        parts: list[str],
        *,
        is_stdout: bool,
    ) -> None:

        nonlocal stdout_count
        nonlocal stderr_count
        nonlocal truncated

        while True:

            chunk = (
                stream.read(
                    4096
                )
            )

            if not chunk:
                break

            with lock:

                count = (
                    stdout_count
                    if is_stdout
                    else stderr_count
                )

                remaining = (
                    MAX_CAPTURE_CHARS
                    - count
                )

                if remaining > 0:

                    captured = (
                        chunk[
                            :remaining
                        ]
                    )

                    parts.append(
                        captured
                    )

                    if is_stdout:

                        stdout_count += len(
                            captured
                        )

                    else:

                        stderr_count += len(
                            captured
                        )

                if (
                    len(
                        chunk
                    )
                    > remaining
                ):

                    truncated = True

    assert (
        process.stdout
        is not None
    )

    assert (
        process.stderr
        is not None
    )

    threads = [
        threading.Thread(
            target=drain,
            args=(
                process.stdout,
                stdout_parts,
            ),
            kwargs={
                "is_stdout":
                    True,
            },
            daemon=True,
        ),

        threading.Thread(
            target=drain,
            args=(
                process.stderr,
                stderr_parts,
            ),
            kwargs={
                "is_stdout":
                    False,
            },
            daemon=True,
        ),
    ]

    for thread in threads:
        thread.start()

    timed_out = False

    try:

        exit_code = (
            process.wait(
                timeout=(
                    timeout_seconds
                )
            )
        )

    except subprocess.TimeoutExpired:

        timed_out = True

        process.kill()

        process.wait()

        exit_code = None

    for thread in threads:

        thread.join(
            timeout=5
        )

    return (
        exit_code,
        "".join(
            stdout_parts
        ),
        "".join(
            stderr_parts
        ),
        timed_out,
        truncated,
    )


def _inspect_image(
    docker: str,
    image: str,
) -> DockerImageIdentity | None:

    result = (
        subprocess.run(
            [
                docker,
                "image",
                "inspect",
                "--format",
                (
                    "{{.Id}}|"
                    "{{join .RepoDigests \",\"}}"
                ),
                image,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=30,
            shell=False,
        )
    )

    if result.returncode != 0:

        return None

    line = (
        result.stdout
        .strip()
    )

    if not line:

        return None

    image_id, _, digest_text = (
        line.partition(
            "|"
        )
    )

    digests = [
        value.strip()

        for value
        in digest_text.split(
            ","
        )

        if value.strip()
    ]

    return (
        DockerImageIdentity(
            requested=image,
            image_id=image_id,
            repo_digests=digests,
        )
    )


def _ensure_image(
    *,
    docker: str,
    image: str,
    allow_pull: bool,
) -> DockerImageIdentity:

    identity = (
        _inspect_image(
            docker,
            image,
        )
    )

    if identity is not None:

        return identity

    if not allow_pull:

        raise RuntimeError(
            "Behavioral evaluation image is not cached: "
            + image
            + ". Re-run with --allow-image-pull "
            "to explicitly authorize the Docker pull."
        )

    result = (
        subprocess.run(
            [
                docker,
                "pull",
                image,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=1800,
            shell=False,
        )
    )

    if result.returncode != 0:

        detail = (
            result.stderr
            or result.stdout
            or "docker pull failed"
        )

        raise RuntimeError(
            "Could not pull behavioral evaluation image: "
            + detail[
                :1000
            ]
        )

    identity = (
        _inspect_image(
            docker,
            image,
        )
    )

    if identity is None:

        raise RuntimeError(
            "Docker image exists after pull but "
            "its identity could not be resolved."
        )

    return identity


def _sandbox_script(
    *,
    record: DeveloperBehavioralHoldoutRecord,
    commands: list[str],
    nonce: str,
) -> str:

    base_commit = (
        shlex.quote(
            record.base_commit
        )
    )

    marker = (
        "__AGENTIC_STAGE_"
        + nonce
        + "__="
    )

    lines = [
        "set -euo pipefail",

        (
            "printf '%s\\n' "
            + shlex.quote(
                marker
                + "base_reset"
            )
        ),

        (
            "git reset --hard "
            + base_commit
        ),

        (
            "test \"$(git rev-parse HEAD)\" = "
            + base_commit
        ),

        (
            "printf '%s\\n' "
            + shlex.quote(
                marker
                + "candidate_patch"
            )
        ),

        (
            "git apply --check "
            "--3way --recount "
            "--ignore-space-change "
            "--whitespace=nowarn "
            "/patches/candidate.diff"
        ),

        (
            "git apply -v "
            "--3way --recount "
            "--ignore-space-change "
            "--whitespace=nowarn "
            "/patches/candidate.diff"
        ),

        "git diff --check",

        (
            "printf '%s\\n' "
            + shlex.quote(
                marker
                + "test_patch"
            )
        ),

        (
            "git apply --check "
            "--3way --recount "
            "--ignore-space-change "
            "--whitespace=nowarn "
            "/patches/test.diff"
        ),

        (
            "git apply -v "
            "--3way --recount "
            "--ignore-space-change "
            "--whitespace=nowarn "
            "/patches/test.diff"
        ),

        (
            "printf '%s\\n' "
            + shlex.quote(
                marker
                + "tests"
            )
        ),
    ]

    lines.extend(
        commands
    )

    lines.append(
        (
            "printf '%s\\n' "
            + shlex.quote(
                marker
                + "complete"
            )
        )
    )

    return "\n".join(
        lines
    )


def _last_stage(
    *,
    stdout: str,
    stderr: str,
    nonce: str,
) -> str | None:

    marker = (
        "__AGENTIC_STAGE_"
        + nonce
        + "__="
    )

    stages = []

    for line in (
        stdout
        + "\n"
        + stderr
    ).splitlines():

        if line.startswith(
            marker
        ):

            stages.append(
                line[
                    len(
                        marker
                    ):
                ].strip()
            )

    if not stages:

        return None

    return stages[
        -1
    ]


def run_behavioral_candidate_patch(
    *,
    behavioral_holdout_directory: Path,
    case_id: str,
    candidate_patch_path: Path,
    allow_execution: bool,
    allow_image_pull: bool = False,
    timeout_seconds: int = 600,
    memory_mb: int = 4096,
    cpus: float = 4.0,
    pids_limit: int = 512,
    output_root: Path = (
        DEFAULT_DEVELOPER_BEHAVIORAL_SANDBOX_ROOT
    ),
) -> DeveloperBehavioralSandboxResult:

    if not allow_execution:

        raise PermissionError(
            "Behavioral sandbox execution requires "
            "explicit allow_execution=True."
        )

    if not (
        1
        <= timeout_seconds
        <= MAX_TIMEOUT_SECONDS
    ):

        raise ValueError(
            "timeout_seconds must be between "
            f"1 and {MAX_TIMEOUT_SECONDS}."
        )

    if not (
        256
        <= memory_mb
        <= MAX_MEMORY_MB
    ):

        raise ValueError(
            "memory_mb must be between "
            f"256 and {MAX_MEMORY_MB}."
        )

    if not (
        0.25
        <= cpus
        <= MAX_CPUS
    ):

        raise ValueError(
            "cpus must be between "
            f"0.25 and {MAX_CPUS}."
        )

    if not (
        32
        <= pids_limit
        <= 4096
    ):

        raise ValueError(
            "pids_limit must be between "
            "32 and 4096."
        )

    (
        manifest,
        records,
    ) = (
        _load_behavioral_holdout(
            behavioral_holdout_directory
        )
    )

    record = (
        _find_case(
            records,
            case_id,
        )
    )

    if (
        _REPOSITORY_PATTERN
        .fullmatch(
            record.repository
        )
        is None
    ):

        raise ValueError(
            "Behavioral case repository identity "
            "is not owner/repo."
        )

    if (
        record.gold_patch_included
        is not False

        or record.evaluation_only
        is not True

        or record.training_eligible
        is not False
    ):

        raise PermissionError(
            "Behavioral case governance flags "
            "are invalid."
        )

    candidate_patch = (
        _read_candidate_patch(
            candidate_patch_path
        )
    )

    test_patch = (
        record.test_patch
    )

    if (
        len(
            test_patch.encode(
                "utf-8"
            )
        )
        > MAX_TEST_PATCH_BYTES
    ):

        raise ValueError(
            "Behavioral test patch exceeds "
            "the evaluator size limit."
        )

    commands = (
        _test_commands(
            record
        )
    )

    docker = (
        _require_docker()
    )

    image_identity = (
        _ensure_image(
            docker=docker,
            image=record.image_name,
            allow_pull=(
                allow_image_pull
            ),
        )
    )

    repository_name = (
        record.repository
        .split(
            "/",
            1,
        )[
            1
        ]
    )

    workdir = (
        "/"
        + repository_name
    )

    run_id = (
        "developer-behavioral-sandbox-"
        + uuid.uuid4().hex
    )

    output_directory = (
        output_root
        .expanduser()
        .resolve()
        / run_id
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    patch_directory = Path(
        tempfile.mkdtemp(
            prefix=".patches-",
            dir=(
                output_directory
            ),
        )
    )

    candidate_path = (
        patch_directory
        / "candidate.diff"
    )

    test_path = (
        patch_directory
        / "test.diff"
    )

    candidate_path.write_text(
        candidate_patch,
        encoding="utf-8",
    )

    test_path.write_text(
        test_patch,
        encoding="utf-8",
    )

    nonce = (
        uuid.uuid4()
        .hex[
            :16
        ]
    )

    script = (
        _sandbox_script(
            record=record,
            commands=commands,
            nonce=nonce,
        )
    )

    container_name = (
        "agentic-behavior-"
        + uuid.uuid4()
        .hex[
            :16
        ]
    )

    docker_command = [
        docker,
        "run",
        "--rm",

        "--name",
        container_name,

        "--network",
        "none",

        "--cap-drop",
        "ALL",

        "--security-opt",
        "no-new-privileges=true",

        "--pids-limit",
        str(
            pids_limit
        ),

        "--memory",
        f"{memory_mb}m",

        "--memory-swap",
        f"{memory_mb}m",

        "--cpus",
        str(
            cpus
        ),

        "--mount",
        (
            "type=bind,"
            "src="
            + str(
                patch_directory
            )
            + ",dst=/patches,"
            "readonly"
        ),

        "--workdir",
        workdir,

        image_identity.image_id,

        "/bin/bash",
        "-lc",
        script,
    ]

    started = (
        time.monotonic()
    )

    try:

        (
            exit_code,
            stdout,
            stderr,
            timed_out,
            truncated,
        ) = (
            _bounded_subprocess(
                docker_command,
                timeout_seconds=(
                    timeout_seconds
                ),
            )
        )

    finally:

        # If the docker client was killed by timeout, the
        # container may still be alive. Cleanup is best-effort.
        if (
            "timed_out"
            in locals()
            and timed_out
        ):

            subprocess.run(
                [
                    docker,
                    "rm",
                    "-f",
                    container_name,
                ],
                capture_output=True,
                check=False,
                timeout=30,
                shell=False,
            )

    duration_ms = int(
        (
            time.monotonic()
            - started
        )
        * 1000
    )

    stage = (
        _last_stage(
            stdout=stdout,
            stderr=stderr,
            nonce=nonce,
        )
    )

    stdout_path = (
        output_directory
        / "stdout.log"
    )

    stderr_path = (
        output_directory
        / "stderr.log"
    )

    stdout_path.write_text(
        stdout,
        encoding="utf-8",
    )

    stderr_path.write_text(
        stderr,
        encoding="utf-8",
    )

    shutil.rmtree(
        patch_directory,
        ignore_errors=True,
    )

    stage_order = {
        None:
            0,

        "base_reset":
            1,

        "candidate_patch":
            2,

        "test_patch":
            3,

        "tests":
            4,

        "complete":
            5,
    }

    observed_stage = (
        stage_order.get(
            stage,
            0,
        )
    )

    report = (
        DeveloperBehavioralSandboxReport(
            run_id=run_id,

            created_at=(
                _utc_now()
            ),

            behavioral_holdout_id=(
                manifest.holdout_id
            ),

            behavioral_records_sha256=(
                manifest.records_sha256
            ),

            record_id=(
                record.record_id
            ),

            source_record_id=(
                record.source_record_id
            ),

            source_identity=(
                record.source_identity
            ),

            repository=(
                record.repository
            ),

            base_commit=(
                record.base_commit
            ),

            image=(
                image_identity
            ),

            candidate_patch_sha256=(
                _sha256_text(
                    candidate_patch
                )
            ),

            test_patch_sha256=(
                _sha256_text(
                    test_patch
                )
            ),

            test_command_count=len(
                commands
            ),

            pids_limit=(
                pids_limit
            ),

            memory_mb=(
                memory_mb
            ),

            cpus=(
                cpus
            ),

            timeout_seconds=(
                timeout_seconds
            ),

            duration_ms=(
                duration_ms
            ),

            container_exit_code=(
                exit_code
            ),

            timed_out=(
                timed_out
            ),

            output_truncated=(
                truncated
            ),

            last_stage=(
                stage
            ),

            candidate_patch_applied=(
                observed_stage
                >= 3
            ),

            test_patch_applied=(
                observed_stage
                >= 4
            ),

            test_execution_started=(
                observed_stage
                >= 4
            ),

            test_process_exit_success=(
                not timed_out
                and exit_code == 0
                and stage == "complete"
            ),

            stdout_sha256=(
                sha256_file(
                    stdout_path
                )
                or ""
            ),

            stderr_sha256=(
                sha256_file(
                    stderr_path
                )
                or ""
            ),

            stdout_path=str(
                stdout_path
            ),

            stderr_path=str(
                stderr_path
            ),

            behavioral_scoring_complete=False,

            promotion_authorized=False,

            output_directory=str(
                output_directory
            ),
        )
    )

    report_path = (
        output_directory
        / "report.json"
    )

    immutable_write_json(
        report_path,
        report,
    )

    return (
        DeveloperBehavioralSandboxResult(
            report=report,

            report_path=str(
                report_path
            ),

            stdout_path=str(
                stdout_path
            ),

            stderr_path=str(
                stderr_path
            ),
        )
    )
