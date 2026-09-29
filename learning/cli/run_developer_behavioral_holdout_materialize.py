from __future__ import annotations

import argparse
import json

from pathlib import Path

from learning.continual.corpus_decontamination import (
    DEFAULT_DECONTAMINATION_PATH,
)

from learning.evaluation.developer_behavioral_holdout import (
    DEFAULT_CORPUS_REGISTRY,
    DEFAULT_DEVELOPER_BEHAVIORAL_HOLDOUT_ROOT,
    materialize_developer_behavioral_holdout,
)


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Recover executable SWE evaluation metadata for "
            "an immutable developer holdout. The parent holdout "
            "defines the exact case identities. Gold patches are "
            "verified but excluded from the behavioral artifact."
        )
    )

    parser.add_argument(
        "--holdout-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--registry",
        type=Path,
        default=(
            DEFAULT_CORPUS_REGISTRY
        ),
    )

    parser.add_argument(
        "--decontamination",
        type=Path,
        default=(
            DEFAULT_DECONTAMINATION_PATH
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            DEFAULT_DEVELOPER_BEHAVIORAL_HOLDOUT_ROOT
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
        materialize_developer_behavioral_holdout(
            holdout_directory=(
                args.holdout_dir
            ),

            registry_path=(
                args.registry
            ),

            decontamination_path=(
                args.decontamination
            ),

            output_root=(
                args.output_root
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

    print()

    print(
        "Developer Behavioral Holdout"
    )

    print(
        "============================"
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
        "Parent       "
        + manifest.parent_holdout_id
    )

    print(
        "Cases        "
        + str(
            manifest.holdout_count
        )
        + "/"
        + str(
            manifest.holdout_count
        )
    )

    print(
        "Identity     "
        + (
            "VERIFIED"
            if manifest.identity_verified
            else "FAILED"
        )
    )

    print(
        "Gold patch   "
        + (
            "INCLUDED"
            if manifest.gold_patch_included
            else "EXCLUDED"
        )
    )

    print(
        "Test patch   "
        + (
            "PRESERVED"
            if manifest.test_patch_included
            else "MISSING"
        )
    )

    print(
        "Transition   FAIL_TO_PASS + PASS_TO_PASS"
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
        "Next gate    sandboxed candidate patch execution"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
