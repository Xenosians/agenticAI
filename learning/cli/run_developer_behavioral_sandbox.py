from __future__ import annotations

import argparse
import json

from pathlib import Path

from learning.evaluation.developer_behavioral_sandbox import (
    behavioral_case_summaries,
    run_behavioral_candidate_patch,
)


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Execute one developer behavioral candidate patch "
            "inside an isolated SWE-rebench Docker image. "
            "This stage captures sandbox execution evidence "
            "but does not yet score FAIL_TO_PASS/PASS_TO_PASS."
        )
    )

    parser.add_argument(
        "--behavioral-holdout-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--list-cases",
        action="store_true",
    )

    parser.add_argument(
        "--case-id",
    )

    parser.add_argument(
        "--candidate-patch",
        type=Path,
    )

    parser.add_argument(
        "--allow-execution",
        action="store_true",
    )

    parser.add_argument(
        "--allow-image-pull",
        action="store_true",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--memory-mb",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--cpus",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--pids-limit",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    return parser


def main() -> int:

    args = (
        build_parser()
        .parse_args()
    )

    if args.list_cases:

        cases = (
            behavioral_case_summaries(
                args.behavioral_holdout_dir
            )
        )

        if args.json:

            print(
                json.dumps(
                    cases,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )

            return 0

        print(
            "Developer Behavioral Cases"
        )

        print(
            "=========================="
        )

        for case in cases:

            print()

            print(
                "Case         "
                + str(
                    case[
                        "case_id"
                    ]
                )
            )

            print(
                "Language     "
                + str(
                    case[
                        "language"
                    ]
                )
            )

            print(
                "Repository   "
                + str(
                    case[
                        "repository"
                    ]
                )
            )

            print(
                "Parser       "
                + str(
                    case[
                        "log_parser"
                    ]
                )
            )

            print(
                "F2P / P2P    "
                + str(
                    case[
                        "fail_to_pass"
                    ]
                )
                + " / "
                + str(
                    case[
                        "pass_to_pass"
                    ]
                )
            )

            print(
                "Commands     "
                + str(
                    case[
                        "test_command_count"
                    ]
                )
            )

            print(
                "Image        "
                + str(
                    case[
                        "image"
                    ]
                )
            )

        return 0

    if not args.case_id:

        raise ValueError(
            "--case-id is required unless "
            "--list-cases is supplied."
        )

    if args.candidate_patch is None:

        raise ValueError(
            "--candidate-patch is required unless "
            "--list-cases is supplied."
        )

    result = (
        run_behavioral_candidate_patch(
            behavioral_holdout_directory=(
                args.behavioral_holdout_dir
            ),

            case_id=(
                args.case_id
            ),

            candidate_patch_path=(
                args.candidate_patch
            ),

            allow_execution=(
                args.allow_execution
            ),

            allow_image_pull=(
                args.allow_image_pull
            ),

            timeout_seconds=(
                args.timeout
            ),

            memory_mb=(
                args.memory_mb
            ),

            cpus=(
                args.cpus
            ),

            pids_limit=(
                args.pids_limit
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
        "Developer Behavioral Sandbox"
    )

    print(
        "============================"
    )

    print(
        "Case         "
        + str(
            report.source_record_id
            or report.record_id
        )
    )

    print(
        "Repository   "
        + report.repository
    )

    print(
        "Network      DISABLED"
    )

    print(
        "Privileges   DROP ALL + NO-NEW-PRIVILEGES"
    )

    print(
        "Candidate    "
        + (
            "APPLIED"
            if report.candidate_patch_applied
            else "NOT APPLIED"
        )
    )

    print(
        "Test patch   "
        + (
            "APPLIED"
            if report.test_patch_applied
            else "NOT APPLIED"
        )
    )

    print(
        "Tests        "
        + (
            "STARTED"
            if report.test_execution_started
            else "NOT STARTED"
        )
    )

    print(
        "Exit         "
        + str(
            report.container_exit_code
        )
    )

    print(
        "Timed out    "
        + str(
            report.timed_out
        ).upper()
    )

    print(
        "Stage        "
        + str(
            report.last_stage
        )
    )

    print(
        "Scoring      NOT RUN"
    )

    print(
        "Promotion    BLOCKED"
    )

    print(
        "Artifact     "
        + report.output_directory
    )

    print()

    print(
        "Next gate    parse benchmark test transitions"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
