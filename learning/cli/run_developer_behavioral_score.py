from __future__ import annotations

import argparse
import json

from pathlib import Path

from learning.evaluation.developer_behavioral_scoring import (
    score_behavioral_sandbox_run,
)


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Score one completed developer behavioral "
            "sandbox run against hidden FAIL_TO_PASS and "
            "PASS_TO_PASS expectations."
        )
    )

    parser.add_argument(
        "--behavioral-holdout-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--sandbox-run-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    args = parser.parse_args()

    result = (
        score_behavioral_sandbox_run(
            behavioral_holdout_directory=(
                args.behavioral_holdout_dir
            ),

            sandbox_run_directory=(
                args.sandbox_run_dir
            ),
        )
    )

    if args.json:

        print(
            json.dumps(
                result.model_dump(
                    mode="json"
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )

        return 0

    report = (
        result.report
    )

    print()

    print(
        "Developer Behavioral Score"
    )

    print(
        "=========================="
    )

    print(
        "Case         "
        + str(
            report.source_record_id
            or report.record_id
        )
    )

    print(
        "Parser       "
        + report.parser_name
    )

    print(
        "Observed     "
        + str(
            report.observed_test_count
        )
        + " tests"
    )

    print(
        "FAIL->PASS   "
        + str(
            len(
                report.resolved_fail_to_pass
            )
        )
        + "/"
        + str(
            len(
                report.expected_fail_to_pass
            )
        )
    )

    print(
        "PASS->PASS   "
        + str(
            len(
                report.preserved_pass_to_pass
            )
        )
        + "/"
        + str(
            len(
                report.expected_pass_to_pass
            )
        )
    )

    print(
        "Logs         "
        + (
            "VERIFIED"
            if report.logs_verified
            else "FAILED"
        )
    )

    print(
        "Scoring      "
        + (
            "COMPLETE"
            if report.scoring_complete
            else "INCOMPLETE"
        )
    )

    print(
        "Resolved     "
        + (
            "YES"
            if report.resolved
            else "NO"
        )
    )

    print(
        "Promotion    BLOCKED"
    )

    print(
        "Artifact     "
        + report.output_directory
    )

    if report.unresolved_fail_to_pass:

        print()

        print(
            "Unresolved FAIL_TO_PASS"
        )

        for name in (
            report.unresolved_fail_to_pass
        ):

            print(
                "  "
                + name
            )

    if report.regressed_pass_to_pass:

        print()

        print(
            "Regressed PASS_TO_PASS"
        )

        for name in (
            report.regressed_pass_to_pass
        ):

            print(
                "  "
                + name
            )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
