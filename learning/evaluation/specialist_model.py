from __future__ import annotations

import hashlib
import json

from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
)

from typing import (
    Any,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from learning.evidence.sanitizer import (
    sanitize_value,
)


class SpecialistModelCaseResult(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True,
            extra="forbid",
        )
    )

    name: str

    user_request: str

    expected_tool: str

    expected_arguments: dict[
        str,
        Any,
    ]

    observed_tool: (
        str
        | None
    ) = None

    observed_arguments: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    raw_output: (
        str
        | None
    ) = None

    parse_error: (
        str
        | None
    ) = None

    passed: bool

    duration_seconds: float

    execution_provenance: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None


class SpecialistModelEvaluationReport(
    BaseModel
):
    """
    Deterministic specialist-model candidate evaluation artifact.

    The artifact is learning/evaluation evidence only. It does not
    authorize provider execution and it does not promote a model.
    """

    model_config = (
        ConfigDict(
            populate_by_name=True,
            extra="forbid",
        )
    )

    schema_name: str = Field(
        default=(
            "specialist-model-evaluation.v1"
        ),
        alias="schema",
    )

    created_at: str

    agent_name: str

    model_key: str

    backend: str

    quantization: str

    compute_dtype: str

    device_map: (
        str
        | None
    ) = None

    total: int

    passed: int

    failed: int

    pass_rate: float

    mean_duration_seconds: (
        float
        | None
    ) = None

    promotion_gate_passed: bool

    cases: list[
        SpecialistModelCaseResult
    ] = Field(
        default_factory=list
    )

    report_sha256: str = ""


def _canonical_json(
    value: Any,
) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
    )


def finalize_specialist_model_report(
    *,
    agent_name: str,
    model_key: str,
    backend: str,
    quantization: str,
    compute_dtype: str,
    device_map: (
        str
        | None
    ),
    cases: list[
        SpecialistModelCaseResult
    ],
) -> SpecialistModelEvaluationReport:
    total = len(
        cases
    )

    passed = sum(
        1

        for case
        in cases

        if case.passed
    )

    failed = (
        total
        - passed
    )

    pass_rate = (
        passed
        / total

        if total
        else 0.0
    )

    mean_duration_seconds = (
        sum(
            case.duration_seconds

            for case
            in cases
        )
        / total

        if total
        else None
    )

    report = (
        SpecialistModelEvaluationReport(
            created_at=(
                datetime
                .now(
                    timezone.utc
                )
                .isoformat()
            ),

            agent_name=(
                agent_name
            ),

            model_key=(
                model_key
            ),

            backend=(
                backend
            ),

            quantization=(
                quantization
            ),

            compute_dtype=(
                compute_dtype
            ),

            device_map=(
                device_map
            ),

            total=(
                total
            ),

            passed=(
                passed
            ),

            failed=(
                failed
            ),

            pass_rate=(
                pass_rate
            ),

            mean_duration_seconds=(
                mean_duration_seconds
            ),

            # Current promotion rule is deliberately strict:
            # every deterministic protocol case must pass.
            promotion_gate_passed=(
                total > 0
                and failed == 0
            ),

            cases=(
                cases
            ),
        )
    )

    payload = (
        report.model_dump(
            mode="json",
            by_alias=True,
        )
    )

    payload[
        "report_sha256"
    ] = ""

    sanitized = (
        sanitize_value(
            payload
        )
    )

    report.report_sha256 = (
        hashlib
        .sha256(
            _canonical_json(
                sanitized
            )
            .encode(
                "utf-8"
            )
        )
        .hexdigest()
    )

    return report


def write_specialist_model_report(
    report: SpecialistModelEvaluationReport,
    path: Path,
) -> Path:
    resolved = (
        path
        .expanduser()
        .resolve()
    )

    resolved.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = (
        sanitize_value(
            report.model_dump(
                mode="json",
                by_alias=True,
            )
        )
    )

    resolved.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return resolved
