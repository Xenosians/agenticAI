from __future__ import annotations

import argparse
import json

from pathlib import Path

from config import Settings

from learning.cli.run_developer_corpus_bridge import (
    _build_token_length_resolver,
    _developer_agent,
    _latest_snapshot,
)

from learning.training.developer_grounded_corpus_bridge import (
    materialize_developer_grounded_sft_snapshot,
)


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Bridge one governed SWE developer snapshot into "
            "mixed repository-context-shaped and issue-only SFT. "
            "No optimizer runs."
        )
    )

    parser.add_argument(
        "--source-id",
        default="swe-rebench-v2",
    )

    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-sequence-tokens",
        type=int,
        default=1536,
    )

    parser.add_argument(
        "--grounded-fraction",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--context-chars",
        type=int,
        default=1400,
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    args = parser.parse_args()

    settings = (
        Settings()
    )

    developer = (
        _developer_agent(
            settings
        )
    )

    profile = (
        settings
        .require_model_profile(
            developer.model
        )
    )

    if profile.model_path is None:

        raise RuntimeError(
            "Developer model profile has no local model_path."
        )

    snapshot_directory = (
        args.snapshot_dir

        if args.snapshot_dir
        is not None

        else _latest_snapshot(
            args.source_id
        )
    )

    target_contract = (
        Path(
            settings.agents_dir
        )
        / "developer-specialist.md"
    )

    token_length_resolver = (
        _build_token_length_resolver(
            model_path=(
                Path(
                    profile.model_path
                )
            ),

            backend=(
                profile.backend
            ),
        )
    )

    result = (
        materialize_developer_grounded_sft_snapshot(
            snapshot_directory=(
                snapshot_directory
            ),

            target_model_key=(
                developer.model
            ),

            base_model_path=(
                Path(
                    profile.model_path
                )
            ),

            target_contract_path=(
                target_contract
            ),

            token_length_resolver=(
                token_length_resolver
            ),

            max_sequence_tokens=(
                args.max_sequence_tokens
            ),

            max_records=(
                args.max_records
            ),

            grounded_fraction=(
                args.grounded_fraction
            ),

            context_chars=(
                args.context_chars
            ),

            max_files=(
                args.max_files
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

    manifest = (
        result.manifest
    )

    print(
        "Developer Grounded Corpus SFT Bridge"
    )

    print(
        "===================================="
    )

    print(
        "materialization="
        + manifest.materialization_id
    )

    print(
        "target="
        + manifest.target_component
    )

    print(
        "model="
        + manifest.target_model_key
    )

    print(
        "evaluation="
        + manifest.evaluation_contract
    )

    print(
        "sft_train="
        + str(
            manifest.sft_train_count
        )
    )

    print(
        "sft_validation="
        + str(
            manifest.sft_validation_count
        )
    )

    print(
        "grounded_selected="
        + str(
            manifest
            .excluded_counts
            .get(
                "grounded_selected",
                0,
            )
        )
    )

    print(
        "issue_only_selected="
        + str(
            manifest
            .excluded_counts
            .get(
                "issue_only_selected",
                0,
            )
        )
    )

    print(
        "sequence_budget="
        + str(
            manifest.sequence_budget_tokens
        )
    )

    print(
        "ready_for_training="
        + str(
            manifest.ready_for_training
        )
    )

    print(
        "training_authorized="
        + str(
            manifest.training_authorized
        )
    )

    print(
        "promotion_authorized="
        + str(
            manifest.promotion_authorized
        )
    )

    print(
        "excluded="
        + json.dumps(
            manifest.excluded_counts,

            sort_keys=True,
        )
    )

    print(
        "output="
        + result.output_directory
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
