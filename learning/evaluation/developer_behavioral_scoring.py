from __future__ import annotations

import hashlib
import re

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from learning.continual.storage import (
    immutable_write_json,
    load_jsonl_models,
    read_json_model,
    sha256_file,
)

from learning.evaluation.developer_behavioral_holdout import (
    DeveloperBehavioralHoldoutManifest,
    DeveloperBehavioralHoldoutRecord,
)

from learning.evaluation.developer_behavioral_sandbox import (
    DeveloperBehavioralSandboxReport,
)

from learning.paths import (
    EVALUATIONS_ROOT,
)


DEFAULT_DEVELOPER_BEHAVIORAL_SCORE_ROOT = (
    EVALUATIONS_ROOT
    / "developer-behavioral-scores"
)


SUPPORTED_LOG_PARSERS = {
    "parse_log_pytest",
    "parse_log_elixir",
}


_TIMING_NORMALIZE_RES = [
    re.compile(
        (
            r"\s*\[\s*\d+(?:\.\d+)?\s*"
            r"(?:ms|s)\s*\]\s*$"
        ),
        re.IGNORECASE,
    ),

    re.compile(
        (
            r"\s+in\s+\d+(?:\.\d+)?\s+"
            r"(?:msec|sec)\b"
        ),
        re.IGNORECASE,
    ),

    re.compile(
        (
            r"\s*\(\s*\d+(?:\.\d+)?\s*"
            r"(?:ms|s)\s*\)\s*$"
        ),
        re.IGNORECASE,
    ),
]


class DeveloperBehavioralScoreReport(
    BaseModel
):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    schema_name: str = Field(
        default=(
            "developer-behavioral-score.v1"
        ),
        alias="schema",
    )

    score_id: str

    created_at: str

    sandbox_run_id: str

    behavioral_holdout_id: str

    behavioral_records_sha256: str

    record_id: str

    source_record_id: (
        str
        | None
    ) = None

    source_identity: str

    candidate_patch_sha256: str

    parser_name: str

    expected_fail_to_pass: list[
        str
    ]

    expected_pass_to_pass: list[
        str
    ]

    observed_test_count: int

    resolved_fail_to_pass: list[
        str
    ]

    unresolved_fail_to_pass: list[
        str
    ]

    preserved_pass_to_pass: list[
        str
    ]

    regressed_pass_to_pass: list[
        str
    ]

    logs_verified: bool

    sandbox_execution_complete: bool

    scoring_complete: bool

    resolved: bool

    promotion_authorized: bool = False

    output_directory: str


class DeveloperBehavioralScoreResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    report: (
        DeveloperBehavioralScoreReport
    )

    report_path: str


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


def normalize_test_name(
    name: str,
) -> str:

    value = name

    for pattern in (
        _TIMING_NORMALIZE_RES
    ):

        value = pattern.sub(
            "",
            value,
        )

    return value.strip()


def parse_log_pytest(
    log: str,
) -> dict[
    str,
    str,
]:

    result: dict[
        str,
        str,
    ] = {}

    statuses = {
        "PASSED",
        "FAILED",
        "ERROR",
        "SKIPPED",
    }

    for line in log.splitlines():

        if not any(
            line.startswith(
                status
            )

            for status
            in statuses
        ):

            continue

        if line.startswith(
            "FAILED"
        ):

            line = line.replace(
                " - ",
                " ",
            )

        parts = (
            line.split()
        )

        if len(
            parts
        ) <= 1:

            continue

        status = (
            parts[
                0
            ]
        )

        test_name = (
            normalize_test_name(
                parts[
                    1
                ]
            )
        )

        if test_name:

            result[
                test_name
            ] = status

    return result


def parse_log_elixir(
    log: str,
) -> dict[
    str,
    str,
]:

    result: dict[
        str,
        str,
    ] = {}

    skipped_re = re.compile(
        (
            r"^\*\s+test\s+(.*?)\s+"
            r"\(skipped\)\s+\[L#\d+\]$"
        )
    )

    passed_timed_re = re.compile(
        (
            r"^\*\s+test\s+(.*?)\s+"
            r"\([0-9]+(?:\.[0-9]+)?ms\)"
            r"\s+\[L#\d+\]$"
        )
    )

    passed_basic_re = re.compile(
        (
            r"^\*\s+test\s+(.*?)\s+"
            r"\[L#\d+\]$"
        )
    )

    failure_header_re = re.compile(
        (
            r"^\d+\)\s+test\s+"
            r"(.*?)\s+\([^)]+\)$"
        )
    )

    for raw in log.splitlines():

        line = raw.strip()

        if not line:
            continue

        match = skipped_re.match(
            line
        )

        if match:

            result[
                normalize_test_name(
                    match.group(
                        1
                    )
                )
            ] = "SKIPPED"

            continue

        match = (
            failure_header_re
            .match(
                line
            )
        )

        if match:

            result[
                normalize_test_name(
                    match.group(
                        1
                    )
                )
            ] = "FAILED"

            continue

        match = (
            passed_timed_re
            .match(
                line
            )
        )

        if match:

            result.setdefault(
                normalize_test_name(
                    match.group(
                        1
                    )
                ),
                "PASSED",
            )

            continue

        match = (
            passed_basic_re
            .match(
                line
            )
        )

        if match:

            result.setdefault(
                normalize_test_name(
                    match.group(
                        1
                    )
                ),
                "PASSED",
            )

    return result


def parse_benchmark_log(
    parser_name: str,
    log: str,
) -> dict[
    str,
    str,
]:

    if (
        parser_name
        == "parse_log_pytest"
    ):

        return (
            parse_log_pytest(
                log
            )
        )

    if (
        parser_name
        == "parse_log_elixir"
    ):

        return (
            parse_log_elixir(
                log
            )
        )

    raise ValueError(
        "Unsupported behavioral log parser: "
        + parser_name
    )


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
        read_json_model(
            manifest_path,
            DeveloperBehavioralHoldoutManifest,
        )
    )

    if (
        sha256_file(
            records_path
        )
        != manifest.records_sha256
    ):

        raise ValueError(
            "Behavioral holdout records SHA-256 mismatch."
        )

    records = (
        load_jsonl_models(
            records_path,
            DeveloperBehavioralHoldoutRecord,
        )
    )

    return (
        manifest,
        records,
    )


def _find_record(
    records: list[
        DeveloperBehavioralHoldoutRecord
    ],
    record_id: str,
) -> DeveloperBehavioralHoldoutRecord:

    for record in records:

        if (
            record.record_id
            == record_id
        ):

            return record

    raise ValueError(
        "Sandbox report references an unknown "
        "behavioral holdout record."
    )


def _score_statuses(
    *,
    statuses: dict[
        str,
        str,
    ],
    fail_to_pass: list[
        str
    ],
    pass_to_pass: list[
        str
    ],
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
]:

    normalized = {
        normalize_test_name(
            name
        ):
            status

        for (
            name,
            status,
        )
        in statuses.items()
    }

    expected_f2p = [
        normalize_test_name(
            value
        )

        for value
        in fail_to_pass
    ]

    expected_p2p = [
        normalize_test_name(
            value
        )

        for value
        in pass_to_pass
    ]

    resolved_f2p = sorted(
        name

        for name
        in expected_f2p

        if (
            normalized.get(
                name
            )
            == "PASSED"
        )
    )

    unresolved_f2p = sorted(
        name

        for name
        in expected_f2p

        if (
            normalized.get(
                name
            )
            != "PASSED"
        )
    )

    preserved_p2p = sorted(
        name

        for name
        in expected_p2p

        if (
            normalized.get(
                name
            )
            == "PASSED"
        )
    )

    regressed_p2p = sorted(
        name

        for name
        in expected_p2p

        if (
            normalized.get(
                name
            )
            != "PASSED"
        )
    )

    return (
        resolved_f2p,
        unresolved_f2p,
        preserved_p2p,
        regressed_p2p,
    )


def score_behavioral_sandbox_run(
    *,
    behavioral_holdout_directory: Path,
    sandbox_run_directory: Path,
    output_root: Path = (
        DEFAULT_DEVELOPER_BEHAVIORAL_SCORE_ROOT
    ),
) -> DeveloperBehavioralScoreResult:

    sandbox_run_directory = (
        sandbox_run_directory
        .expanduser()
        .resolve()
    )

    sandbox_report_path = (
        sandbox_run_directory
        / "report.json"
    )

    if not sandbox_report_path.is_file():

        raise ValueError(
            "Sandbox report does not exist."
        )

    sandbox = (
        read_json_model(
            sandbox_report_path,
            DeveloperBehavioralSandboxReport,
        )
    )

    (
        holdout,
        records,
    ) = (
        _load_behavioral_holdout(
            behavioral_holdout_directory
        )
    )

    if (
        sandbox.behavioral_holdout_id
        != holdout.holdout_id
    ):

        raise ValueError(
            "Sandbox report belongs to a different "
            "behavioral holdout."
        )

    if (
        sandbox.behavioral_records_sha256
        != holdout.records_sha256
    ):

        raise ValueError(
            "Sandbox report behavioral records "
            "SHA-256 mismatch."
        )

    record = (
        _find_record(
            records,
            sandbox.record_id,
        )
    )

    if (
        sandbox.source_identity
        != record.source_identity
    ):

        raise ValueError(
            "Sandbox source identity mismatch."
        )

    expected_test_patch_sha = (
        _sha256_text(
            record.test_patch
        )
    )

    if (
        sandbox.test_patch_sha256
        != expected_test_patch_sha
    ):

        raise ValueError(
            "Sandbox test patch provenance mismatch."
        )

    parser_name = (
        record.install_config.get(
            "log_parser"
        )
    )

    if not isinstance(
        parser_name,
        str,
    ):

        raise ValueError(
            "Behavioral case does not define "
            "install_config.log_parser."
        )

    parser_name = (
        parser_name.strip()
    )

    if (
        parser_name
        not in SUPPORTED_LOG_PARSERS
    ):

        raise ValueError(
            "Behavioral case uses unsupported parser: "
            + parser_name
        )

    stdout_path = Path(
        sandbox.stdout_path
    )

    stderr_path = Path(
        sandbox.stderr_path
    )

    if not (
        stdout_path.is_file()
        and stderr_path.is_file()
    ):

        raise ValueError(
            "Sandbox log files are missing."
        )

    stdout_verified = (
        sha256_file(
            stdout_path
        )
        == sandbox.stdout_sha256
    )

    stderr_verified = (
        sha256_file(
            stderr_path
        )
        == sandbox.stderr_sha256
    )

    logs_verified = (
        stdout_verified
        and stderr_verified
    )

    if not logs_verified:

        raise ValueError(
            "Sandbox log SHA-256 verification failed."
        )

    stdout = (
        stdout_path.read_text(
            encoding="utf-8"
        )
    )

    stderr = (
        stderr_path.read_text(
            encoding="utf-8"
        )
    )

    statuses = (
        parse_benchmark_log(
            parser_name,
            (
                stdout
                + "\n"
                + stderr
            ),
        )
    )

    (
        resolved_f2p,
        unresolved_f2p,
        preserved_p2p,
        regressed_p2p,
    ) = (
        _score_statuses(
            statuses=statuses,
            fail_to_pass=(
                record.fail_to_pass
            ),
            pass_to_pass=(
                record.pass_to_pass
            ),
        )
    )

    sandbox_execution_complete = (
        sandbox.test_execution_started
        and not sandbox.timed_out
        and not sandbox.output_truncated
    )

    scoring_complete = (
        logs_verified
        and sandbox_execution_complete
    )

    resolved = (
        scoring_complete
        and not unresolved_f2p
        and not regressed_p2p
    )

    identity = {
        "sandbox_run_id":
            sandbox.run_id,

        "behavioral_holdout_id":
            holdout.holdout_id,

        "behavioral_records_sha256":
            holdout.records_sha256,

        "record_id":
            record.record_id,

        "candidate_patch_sha256":
            sandbox.candidate_patch_sha256,

        "stdout_sha256":
            sandbox.stdout_sha256,

        "stderr_sha256":
            sandbox.stderr_sha256,

        "parser_name":
            parser_name,
    }

    score_id = (
        "developer-behavioral-score-"
        + _sha256_text(
            str(
                sorted(
                    identity.items()
                )
            )
        )[:24]
    )

    output_directory = (
        output_root
        .expanduser()
        .resolve()
        / score_id
    )

    report_path = (
        output_directory
        / "report.json"
    )

    if report_path.exists():

        report = (
            read_json_model(
                report_path,
                DeveloperBehavioralScoreReport,
            )
        )

        return (
            DeveloperBehavioralScoreResult(
                report=report,
                report_path=str(
                    report_path
                ),
            )
        )

    report = (
        DeveloperBehavioralScoreReport(
            score_id=score_id,

            created_at=(
                _utc_now()
            ),

            sandbox_run_id=(
                sandbox.run_id
            ),

            behavioral_holdout_id=(
                holdout.holdout_id
            ),

            behavioral_records_sha256=(
                holdout.records_sha256
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

            candidate_patch_sha256=(
                sandbox.candidate_patch_sha256
            ),

            parser_name=(
                parser_name
            ),

            expected_fail_to_pass=sorted(
                normalize_test_name(
                    value
                )

                for value
                in record.fail_to_pass
            ),

            expected_pass_to_pass=sorted(
                normalize_test_name(
                    value
                )

                for value
                in record.pass_to_pass
            ),

            observed_test_count=len(
                statuses
            ),

            resolved_fail_to_pass=(
                resolved_f2p
            ),

            unresolved_fail_to_pass=(
                unresolved_f2p
            ),

            preserved_pass_to_pass=(
                preserved_p2p
            ),

            regressed_pass_to_pass=(
                regressed_p2p
            ),

            logs_verified=(
                logs_verified
            ),

            sandbox_execution_complete=(
                sandbox_execution_complete
            ),

            scoring_complete=(
                scoring_complete
            ),

            resolved=(
                resolved
            ),

            promotion_authorized=False,

            output_directory=str(
                output_directory
            ),
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    immutable_write_json(
        report_path,
        report,
    )

    return (
        DeveloperBehavioralScoreResult(
            report=report,
            report_path=str(
                report_path
            ),
        )
    )
