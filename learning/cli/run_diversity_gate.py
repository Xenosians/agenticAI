from __future__ import annotations

import argparse

from pathlib import (
    Path,
)

from learning.curation.engine import (
    curate_corpus,
)

from learning.curation.diversity_gate import (
    DiversityGatePolicy,
    evaluate_diversity_gate,
)


PROJECT_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        1
    ]
)


DEFAULT_TRAJECTORIES = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "trajectories.jsonl"
)


DEFAULT_CORRECTIONS = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "corrections.jsonl"
)


DEFAULT_REVIEWS = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "reviews.jsonl"
)


DEFAULT_EVAL_DIRECTORY = (
    PROJECT_ROOT
    / "learning"
    / "evals"
)


def build_parser(
) -> argparse.ArgumentParser:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Evaluate trusted curated runtime learning "
                "evidence for minimum diversity and balance "
                "before dataset export."
            )
        )
    )

    parser.add_argument(
        "--trajectories",
        type=Path,
        default=(
            DEFAULT_TRAJECTORIES
        ),
    )

    parser.add_argument(
        "--corrections",
        type=Path,
        default=(
            DEFAULT_CORRECTIONS
        ),
    )

    parser.add_argument(
        "--reviews",
        type=Path,
        default=(
            DEFAULT_REVIEWS
        ),
        help=(
            "Append-only trusted review ledger. "
            "Defaults to "
            ".runtime/learning/reviews.jsonl."
        ),
    )

    parser.add_argument(
        "--eval",
        dest="eval_paths",
        type=Path,
        action="append",
        default=None,
    )

    parser.add_argument(
        "--min-eligible-trajectories",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--min-unique-requests",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--min-unique-behavior-patterns",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--min-unique-domains",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--min-unique-capabilities",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-duplicate-request-rate",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--max-dominant-domain-share",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--max-dominant-capability-share",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
    )

    return parser


def _validate_non_negative(
    *,
    name: str,
    value: int,
) -> None:

    if value < 0:

        raise ValueError(
            f"{name} must be >= 0."
        )


def _validate_rate(
    *,
    name: str,
    value: float,
) -> None:

    if (
        value < 0.0
        or value > 1.0
    ):

        raise ValueError(
            f"{name} must be between "
            "0.0 and 1.0."
        )


def _build_policy(
    args,
) -> DiversityGatePolicy:

    _validate_non_negative(
        name="min_eligible_trajectories",
        value=args.min_eligible_trajectories,
    )

    _validate_non_negative(
        name="min_unique_requests",
        value=args.min_unique_requests,
    )

    _validate_non_negative(
        name="min_unique_behavior_patterns",
        value=args.min_unique_behavior_patterns,
    )

    _validate_non_negative(
        name="min_unique_domains",
        value=args.min_unique_domains,
    )

    _validate_non_negative(
        name="min_unique_capabilities",
        value=args.min_unique_capabilities,
    )

    _validate_rate(
        name="max_duplicate_request_rate",
        value=args.max_duplicate_request_rate,
    )

    _validate_rate(
        name="max_dominant_domain_share",
        value=args.max_dominant_domain_share,
    )

    _validate_rate(
        name="max_dominant_capability_share",
        value=args.max_dominant_capability_share,
    )

    return (
        DiversityGatePolicy(
            min_eligible_trajectories=(
                args.min_eligible_trajectories
            ),

            min_unique_requests=(
                args.min_unique_requests
            ),

            min_unique_behavior_patterns=(
                args.min_unique_behavior_patterns
            ),

            min_unique_domains=(
                args.min_unique_domains
            ),

            min_unique_capabilities=(
                args.min_unique_capabilities
            ),

            max_duplicate_request_rate=(
                args.max_duplicate_request_rate
            ),

            max_dominant_domain_share=(
                args.max_dominant_domain_share
            ),

            max_dominant_capability_share=(
                args.max_dominant_capability_share
            ),
        )
    )


def main(
) -> int:

    args = (
        build_parser()
        .parse_args()
    )

    try:

        if (
            args.eval_paths
            is None
        ):

            eval_paths = (
                sorted(
                    DEFAULT_EVAL_DIRECTORY
                    .glob(
                        "*.jsonl"
                    )
                )
            )

        else:

            eval_paths = (
                args.eval_paths
            )

        policy = (
            _build_policy(
                args
            )
        )

        curation_report = (
            curate_corpus(
                trajectory_path=(
                    args.trajectories
                ),

                correction_path=(
                    args.corrections
                ),

                review_path=(
                    args.reviews
                ),

                eval_paths=(
                    eval_paths
                ),
            )
        )

        gate_report = (
            evaluate_diversity_gate(
                trajectory_path=(
                    args.trajectories
                ),

                curation_report=(
                    curation_report
                ),

                policy=(
                    policy
                ),
            )
        )

    except Exception as exc:

        print(
            "ERROR: "
            f"{exc}"
        )

        return 1

    if args.json:

        print(
            gate_report
            .model_dump_json(
                by_alias=True,
                indent=2,
            )
        )

    else:

        metrics = (
            gate_report.metrics
        )

        print(
            "Agentic AI Diversity / Balance Gate"
        )

        print(
            "==================================="
        )

        print(
            f"Trusted reviews:         "
            f"{curation_report.review_count}"
        )

        print(
            f"Curated eligible:        "
            f"{metrics.eligible_trajectory_count}"
        )

        print(
            f"Unique requests:         "
            f"{metrics.unique_request_count}"
        )

        print(
            f"Duplicate requests:      "
            f"{metrics.duplicate_request_count}"
        )

        print(
            f"Duplicate rate:          "
            f"{metrics.duplicate_request_rate * 100:.1f}%"
        )

        print(
            f"Behavior patterns:       "
            f"{metrics.unique_behavior_pattern_count}"
        )

        print(
            f"Unique domains:          "
            f"{metrics.unique_domain_count}"
        )

        print(
            f"Unique capabilities:     "
            f"{metrics.unique_capability_count}"
        )

        print(
            f"Dominant domain:         "
            f"{metrics.dominant_domain or '(none)'}"
        )

        print(
            f"Dominant domain share:   "
            f"{metrics.dominant_domain_share * 100:.1f}%"
        )

        print(
            f"Dominant capability:     "
            f"{metrics.dominant_capability or '(none)'}"
        )

        print(
            f"Dominant capability share: "
            f"{metrics.dominant_capability_share * 100:.1f}%"
        )

        print(
            "\n"
            "Gate Checks"
        )

        print(
            "-----------"
        )

        for check in (
            gate_report.checks
        ):

            state = (
                "PASS"
                if check.passed
                else "FAIL"
            )

            print(
                f"[{state}] "
                f"{check.name}: "
                f"{check.actual} "
                f"{check.comparator} "
                f"{check.threshold}"
            )

        print(
            "\n"
            "Promotion Readiness"
        )

        print(
            "-------------------"
        )

        if (
            gate_report
            .promotion_eligible
        ):

            print(
                "PASS"
            )

        else:

            print(
                "FAIL"
            )

            for name in (
                gate_report
                .failed_checks
            ):

                print(
                    f"  {name}"
                )

    if (
        args.fail_on_gate
        and not gate_report
        .promotion_eligible
    ):

        return 2

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )