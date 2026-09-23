from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from learning.training.jira_sft import (  # noqa: E402
    DEFAULT_TRAINED_MODEL_KEY,
    register_trained_candidate_profile,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/c/project/agenticaiPersonal/Models/BLOOMZ-560M-Jira-SFT"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Register a verified merged Jira SFT checkpoint as a "
            "logical model profile without promoting production Jira."
        )
    )

    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--model-key",
        default=DEFAULT_TRAINED_MODEL_KEY,
    )

    args = parser.parse_args()

    training_manifest = (
        args.training_manifest
    )

    if training_manifest is None:
        latest = (
            args.output_root
            / "latest-training.json"
        )

        if not latest.is_file():
            raise SystemExit(
                "No latest-training.json exists. Run Jira SFT first "
                "or pass --training-manifest explicitly."
            )

        payload = json.loads(
            latest.read_text(
                encoding="utf-8"
            )
        )

        training_manifest = Path(
            payload[
                "training_manifest"
            ]
        )

    profile = register_trained_candidate_profile(
        env_path=(
            PROJECT_ROOT
            / ".env"
        ),
        training_manifest_path=(
            training_manifest
        ),
        model_key=args.model_key,
    )

    print(
        "JIRA TRAINED CANDIDATE REGISTRATION: PASS"
    )
    print(
        "model key:",
        args.model_key,
    )
    print(
        "profile:",
        profile,
    )
    print()
    print(
        "jira-specialist.md was NOT modified."
    )
    print(
        "Production Jira remains on its current model until an explicit "
        "post-evaluation promotion decision."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
