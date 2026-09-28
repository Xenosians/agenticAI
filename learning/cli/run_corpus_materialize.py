from __future__ import annotations

import argparse
import json

from pathlib import Path

from learning.continual.corpus_materializer import (
    materialize_corpus_source,
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
            "Stream and materialize one bounded "
            "continual-learning corpus snapshot."
        )
    )

    parser.add_argument(
        "source_id",
        help=(
            "source_id from "
            "config/continual_corpus_registry.json"
        ),
    )

    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
    )

    parser.add_argument(
        "--scan-limit",
        type=int,
        default=None,
        help=(
            "Maximum upstream rows inspected during this run."
        ),
    )

    parser.add_argument(
        "--reset-cursor",
        action="store_true",
        help=(
            "Start from upstream index 0 without deleting "
            "previous immutable snapshots."
        ),
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

    result = (
        materialize_corpus_source(
            registry_path=(
                args.registry
            ),
            source_id=(
                args.source_id
            ),
            scan_limit=(
                args.scan_limit
            ),
            reset_cursor=(
                args.reset_cursor
            ),
        )
    )

    payload = (
        result.model_dump(
            mode="json"
        )
    )

    if args.json:

        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )

        return 0

    manifest = result.manifest

    print(
        "Continual Corpus Materialization"
    )

    print(
        "================================"
    )

    print(
        "source_id="
        + manifest.source_id
    )

    print(
        "snapshot_id="
        + manifest.snapshot_id
    )

    print(
        "cursor="
        + str(
            manifest.cursor_start
        )
        + "->"
        + str(
            manifest.cursor_end
        )
    )

    print(
        "scanned="
        + str(
            manifest.scanned_count
        )
    )

    print(
        "accepted="
        + str(
            manifest.accepted_count
        )
    )

    print(
        "training_eligible="
        + str(
            manifest.training_eligible
        )
    )

    print(
        "training_blockers="
        + json.dumps(
            manifest.training_blockers,
            sort_keys=True,
        )
    )

    print(
        "output="
        + manifest.output_directory
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
