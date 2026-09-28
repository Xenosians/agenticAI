from __future__ import annotations

import argparse
import json

from pathlib import Path

from learning.continual.corpus_materializer import (
    CorpusCursorStore,
    list_corpus_snapshots,
    snapshot_training_blockers,
)

from learning.continual.corpus_registry import (
    load_corpus_registry,
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
            "Inspect continual corpus registry, "
            "streaming cursors and immutable snapshots."
        )
    )

    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    return parser


def build_snapshot(
    registry_path: Path,
) -> dict:

    registry = (
        load_corpus_registry(
            registry_path
        )
    )

    cursor_store = (
        CorpusCursorStore()
    )

    snapshots = (
        list_corpus_snapshots()
    )

    latest_by_source = {}

    for snapshot in snapshots:

        latest_by_source[
            snapshot.source_id
        ] = snapshot

    sources = []

    for source in registry.sources:

        latest = (
            latest_by_source.get(
                source.source_id
            )
        )

        sources.append(
            {
                "source_id":
                    source.source_id,

                "provider":
                    source.provider,

                "dataset_id":
                    source.dataset_id,

                "enabled":
                    source.enabled,

                "registry_training_eligible":
                    source.training_eligible,

                "target_component":
                    source.target_component,

                "objectives":
                    list(
                        source.objectives
                    ),

                "cursor":
                    cursor_store.get(
                        source
                    ),

                "current_blockers":
                    snapshot_training_blockers(
                        source
                    ),

                "latest_snapshot":
                    (
                        latest.model_dump(
                            mode="json"
                        )
                        if latest
                        is not None
                        else None
                    ),
            }
        )

    return {
        "schema":
            "continual-corpus-status.v1",

        "registry":
            str(
                registry_path
                .expanduser()
                .resolve()
            ),

        "mix":
            registry.mix.model_dump(
                mode="json"
            ),

        "source_count":
            len(
                sources
            ),

        "enabled_source_count":
            sum(
                1
                for source
                in sources
                if source[
                    "enabled"
                ]
            ),

        "snapshot_count":
            len(
                snapshots
            ),

        "sources":
            sources,
    }


def main() -> int:

    args = (
        build_parser()
        .parse_args()
    )

    snapshot = (
        build_snapshot(
            args.registry
        )
    )

    if args.json:

        print(
            json.dumps(
                snapshot,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )

        return 0

    print(
        "Continual Corpus Status"
    )

    print(
        "======================="
    )

    print(
        "sources="
        + str(
            snapshot[
                "source_count"
            ]
        )
    )

    print(
        "enabled="
        + str(
            snapshot[
                "enabled_source_count"
            ]
        )
    )

    print(
        "snapshots="
        + str(
            snapshot[
                "snapshot_count"
            ]
        )
    )

    print()

    for source in snapshot[
        "sources"
    ]:

        print(
            source[
                "source_id"
            ]
        )

        print(
            "  provider="
            + str(
                source[
                    "provider"
                ]
            )
        )

        print(
            "  enabled="
            + str(
                source[
                    "enabled"
                ]
            )
        )

        print(
            "  cursor="
            + str(
                source[
                    "cursor"
                ]
            )
        )

        print(
            "  objectives="
            + json.dumps(
                source[
                    "objectives"
                ]
            )
        )

        print(
            "  blockers="
            + json.dumps(
                source[
                    "current_blockers"
                ]
            )
        )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
