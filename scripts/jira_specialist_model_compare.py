from __future__ import annotations

import argparse
import json
import sys

from pathlib import (
    Path,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def _load_report(
    path: Path,
) -> dict:
    resolved = (
        path
        .expanduser()
        .resolve()
    )

    payload = (
        json.loads(
            resolved.read_text(
                encoding="utf-8"
            )
        )
    )

    if (
        not isinstance(
            payload,
            dict,
        )
        or payload.get(
            "schema"
        )
        != "specialist-model-evaluation.v1"
    ):
        raise ValueError(
            f"Not a specialist-model evaluation report: {resolved}"
        )

    return payload


def _case_map(
    report: dict,
) -> dict[str, dict]:
    return {
        item[
            "name"
        ]:
            item

        for item
        in report.get(
            "cases",
            [],
        )

        if isinstance(
            item,
            dict,
        )
        and isinstance(
            item.get(
                "name"
            ),
            str,
        )
    }


def main(
) -> int:
    parser = (
        argparse.ArgumentParser(
            description=(
                "Compare two Jira specialist model evaluation reports."
            )
        )
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        default=(
            PROJECT_ROOT
            / ".runtime"
            / "evaluation"
            / "jira_specialist_hub-main.json"
        ),
    )

    parser.add_argument(
        "--candidate",
        type=Path,
        default=(
            PROJECT_ROOT
            / ".runtime"
            / "evaluation"
            / "jira_specialist_jira-func.json"
        ),
    )

    args = (
        parser.parse_args()
    )

    try:
        baseline = _load_report(
            args.baseline
        )

        candidate = _load_report(
            args.candidate
        )

    except Exception as exc:
        print(
            "Comparison failed:",
            repr(exc),
        )
        return 2

    if (
        baseline.get(
            "agent_name"
        )
        != candidate.get(
            "agent_name"
        )
    ):
        print(
            "Comparison failed: reports target different agents."
        )
        return 2

    baseline_cases = (
        _case_map(
            baseline
        )
    )

    candidate_cases = (
        _case_map(
            candidate
        )
    )

    if (
        set(
            baseline_cases
        )
        != set(
            candidate_cases
        )
    ):
        print(
            "Comparison failed: reports do not contain the same case set."
        )
        return 2

    regressions: list[str] = []
    improvements: list[str] = []

    for name in sorted(
        baseline_cases
    ):
        baseline_passed = bool(
            baseline_cases[
                name
            ]
            .get(
                "passed",
                False,
            )
        )

        candidate_passed = bool(
            candidate_cases[
                name
            ]
            .get(
                "passed",
                False,
            )
        )

        if (
            baseline_passed
            and not candidate_passed
        ):
            regressions.append(
                name
            )

        if (
            not baseline_passed
            and candidate_passed
        ):
            improvements.append(
                name
            )

    baseline_rate = float(
        baseline.get(
            "pass_rate",
            0.0,
        )
    )

    candidate_rate = float(
        candidate.get(
            "pass_rate",
            0.0,
        )
    )

    baseline_duration = (
        baseline.get(
            "mean_duration_seconds"
        )
    )

    candidate_duration = (
        candidate.get(
            "mean_duration_seconds"
        )
    )

    promotion_eligible = (
        bool(
            candidate.get(
                "promotion_gate_passed",
                False,
            )
        )
        and not regressions
        and candidate_rate
        >= baseline_rate
    )

    print(
        "JIRA SPECIALIST MODEL COMPARISON"
    )
    print(
        "================================"
    )
    print(
        "baseline:",
        baseline.get(
            "model_key"
        ),
    )
    print(
        "candidate:",
        candidate.get(
            "model_key"
        ),
    )
    print(
        "baseline pass rate:",
        f"{baseline_rate:.3f}",
    )
    print(
        "candidate pass rate:",
        f"{candidate_rate:.3f}",
    )
    print(
        "pass-rate delta:",
        f"{candidate_rate - baseline_rate:+.3f}",
    )
    print(
        "baseline mean seconds:",
        baseline_duration,
    )
    print(
        "candidate mean seconds:",
        candidate_duration,
    )
    print(
        "regressions:",
        regressions,
    )
    print(
        "improvements:",
        improvements,
    )
    print(
        "candidate protocol gate:",
        candidate.get(
            "promotion_gate_passed"
        ),
    )
    print(
        "promotion comparison eligible:",
        promotion_eligible,
    )
    print()
    print(
        "This comparison does not modify jira-specialist.md or "
        "promote a model automatically."
    )

    return (
        0
        if promotion_eligible
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
