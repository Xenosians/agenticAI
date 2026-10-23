from __future__ import annotations

import argparse
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
    build_jira_sft_corpus,
    verify_jira_sft_corpus,
)


DEFAULT_CORPUS = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "jira-sft"
    / "v1"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify the deterministic Jira specialist SFT seed corpus."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CORPUS,
    )

    parser.add_argument(
        "--worker-prompt-profile",
        choices=("standard", "compact", "compact_grounded"),
        default=None,
        help=(
            "Explicit candidate prompt contract; does not change the serving profile. "
            "Omit to bind the configured base model profile."
        ),
    )
    parser.add_argument(
        "--prompt-layout", choices=("legacy", "runtime_task"), default="legacy",
        help="runtime_task pins per-task capability prompts with the original request last.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace an existing corpus directory after rebuilding "
            "identity hashes from the current Jira contract."
        ),
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.verify_only:
        (
            manifest,
            train,
            validation,
            _system_prompt,
        ) = verify_jira_sft_corpus(
            project_root=PROJECT_ROOT,
            corpus_directory=args.output,
            worker_prompt_profile=args.worker_prompt_profile,
        )

    else:
        manifest = build_jira_sft_corpus(
            project_root=PROJECT_ROOT,
            output_directory=args.output,
            force=args.force,
            worker_prompt_profile=args.worker_prompt_profile,
            prompt_layout=args.prompt_layout,
        )

        (
            manifest,
            train,
            validation,
            _system_prompt,
        ) = verify_jira_sft_corpus(
            project_root=PROJECT_ROOT,
            corpus_directory=args.output,
            worker_prompt_profile=args.worker_prompt_profile,
        )

    print(
        "JIRA SFT CORPUS: PASS"
    )
    print(
        "directory:",
        args.output.resolve(),
    )
    print(
        "records:",
        manifest.record_count,
    )
    print(
        "train:",
        len(train),
    )
    print(
        "validation:",
        len(validation),
    )
    print(
        "prompt profile:",
        manifest.worker_prompt_profile,
    )
    print("prompt layout:", manifest.prompt_layout)
    print(
        "train sha256:",
        manifest.train_sha256,
    )
    print(
        "validation sha256:",
        manifest.validation_sha256,
    )
    print(
        "per-tool counts:",
        manifest.per_tool_counts,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
