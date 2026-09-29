from __future__ import annotations

import argparse
import json

from pathlib import Path

from learning.cli.crashlog import (
    run_with_crashlog,
)

from config import (
    Settings,
)

from learning.cli.reporting import (
    render_developer_holdout_summary,
)

from learning.continual.checkpoints import (
    AdapterCheckpointStore,
)

from learning.evaluation.developer_holdout_eval import (
    DEFAULT_DEVELOPER_HOLDOUT_EVAL_ROOT,
    evaluate_developer_holdout,
)


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Compare a registered developer candidate "
            "against its frozen base on an immutable "
            "reserved holdout. This command never trains, "
            "activates, or promotes a checkpoint."
        )
    )

    parser.add_argument(
        "--checkpoint-id",
        required=True,
    )

    parser.add_argument(
        "--holdout-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            "Forward-only heldout evaluation sequence budget. "
            "May be larger than the original training budget. "
            "Gold completions are never truncated."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            DEFAULT_DEVELOPER_HOLDOUT_EVAL_ROOT
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

    store = (
        AdapterCheckpointStore()
    )

    checkpoint = (
        store.verify_adapter(
            args.checkpoint_id
        )
    )

    settings = (
        Settings()
    )

    profile = (
        settings
        .require_model_profile(
            checkpoint
            .target_model_key
        )
    )

    report = (
        evaluate_developer_holdout(
            checkpoint_id=(
                checkpoint
                .checkpoint_id
            ),

            holdout_directory=(
                args.holdout_dir
            ),

            backend=(
                profile.backend
            ),

            max_sequence_tokens=(
                args.max_length
            ),

            output_root=(
                args.output_root
            ),
        )
    )

    if args.json:

        print(
            json.dumps(
                report.model_dump(
                    mode="json"
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )

    else:

        print(
            render_developer_holdout_summary(
                report
            )
        )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        run_with_crashlog(
            "developer-holdout-eval",
            main,
        )
    )
