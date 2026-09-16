from __future__ import annotations

import json

from pathlib import (
    Path,
)

from learning.evaluation.eval_types import (
    EvaluationCase,
)


def load_evaluation_cases(
    path: Path,
) -> list[
    EvaluationCase
]:
    """
    Load a version-controlled JSONL evaluation suite.

    Duplicate case IDs are rejected because evaluation provenance
    must remain unambiguous.
    """

    resolved = (
        path
        .expanduser()
        .resolve()
    )

    if not resolved.is_file():
        raise ValueError(
            "Evaluation suite does not "
            f"exist: {resolved}"
        )

    cases: list[
        EvaluationCase
    ] = []

    seen_ids: set[
        str
    ] = set()

    for (
        line_number,
        line,
    ) in enumerate(
        resolved
        .read_text(
            encoding="utf-8"
        )
        .splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            raw = (
                json.loads(
                    line
                )
            )

        except Exception as exc:
            raise ValueError(
                (
                    "Invalid evaluation JSON "
                    f"at line {line_number}: "
                    f"{exc}"
                )
            ) from exc

        try:
            case = (
                EvaluationCase
                .model_validate(
                    raw
                )
            )

        except Exception as exc:
            raise ValueError(
                (
                    "Invalid evaluation case "
                    f"at line {line_number}: "
                    f"{exc}"
                )
            ) from exc

        case_id = (
            case.case_id
            .strip()
        )

        if not case_id:
            raise ValueError(
                (
                    "Evaluation case at line "
                    f"{line_number} has an "
                    "empty case_id."
                )
            )

        if case_id in seen_ids:
            raise ValueError(
                (
                    "Duplicate evaluation "
                    f"case_id: {case_id}"
                )
            )

        seen_ids.add(
            case_id
        )

        cases.append(
            case
        )

    if not cases:
        raise ValueError(
            "Evaluation suite contains "
            "no cases."
        )

    return cases