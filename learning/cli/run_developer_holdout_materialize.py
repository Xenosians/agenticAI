from __future__ import annotations

import argparse
import json

from pathlib import Path

from learning.continual.corpus_decontamination import (
    DEFAULT_DECONTAMINATION_PATH,
)

from learning.evaluation.developer_holdout import (
    DEFAULT_DEVELOPER_HOLDOUT_ROOT,
    materialize_developer_holdout,
)

from learning.paths import (
    REPOSITORY_ROOT,
)


DEFAULT_REGISTRY = (
    REPOSITORY_ROOT
    / "config"
    / "continual_corpus_registry.json"
)


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Materialize the deterministic reserved "
            "developer holdout partition. "
            "The resulting artifact is evaluation-only "
            "and can never authorize training."
        )
    )

    parser.add_argument(
        "source_id",
    )

    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
    )

    parser.add_argument(
        "--decontamination",
        type=Path,
        default=(
            DEFAULT_DECONTAMINATION_PATH
        ),
    )

    parser.add_argument(
        "--scan-start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--scan-limit",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            DEFAULT_DEVELOPER_HOLDOUT_ROOT
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print the complete machine-readable "
            "materialization artifact."
        ),
    )

    return parser


def main() -> int:

    args = (
        build_parser()
        .parse_args()
    )

    result = (
        materialize_developer_holdout(
            registry_path=(
                args.registry
            ),
            source_id=(
                args.source_id
            ),
            scan_start=(
                args.scan_start
            ),
            scan_limit=(
                args.scan_limit
            ),
            max_records=(
                args.max_records
            ),
            output_root=(
                args.output_root
            ),
            decontamination_path=(
                args.decontamination
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

    manifest = result.manifest

    print()
    print(
        "Developer Holdout Materialization"
    )

    print(
        "================================="
    )

    print(
        "Status       READY (evaluation only)"
    )

    print(
        "Source       "
        + manifest.source_id
    )

    print(
        "Revision     "
        + str(
            manifest.revision
        )
    )

    print(
        "Scanned      "
        + str(
            manifest.scanned_count
        )
        + " ("
        + str(
            manifest.scan_start
        )
        + " -> "
        + str(
            manifest.scan_end
        )
        + ")"
    )

    print(
        "Heldout      "
        + str(
            manifest.holdout_count
        )
    )

    print(
        "Training     BLOCKED"
    )

    print(
        "Promotion    BLOCKED"
    )

    print(
        "Artifact     "
        + manifest.output_directory
    )

    print()
    print(
        "Next gate    independent developer "
        "holdout evaluation"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
