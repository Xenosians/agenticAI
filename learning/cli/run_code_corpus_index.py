from __future__ import annotations

import argparse
import json
from pathlib import Path

from learning.code_corpus.indexer import (
    DEFAULT_CODE_CORPUS_ROOT,
    build_code_corpus,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one provenance-pinned semantic code corpus. "
            "Source code is split semantically before tokenizer encoding. "
            "This command does NOT train or activate a model."
        )
    )

    parser.add_argument(
        "--repository",
        required=True,
        help=(
            "Logical corpus/repository id such as ai, backend, frontend, "
            "or general-python."
        ),
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--source-kind",
        choices=[
            "general",
            "project",
        ],
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_CODE_CORPUS_ROOT,
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=8000,
    )

    parser.add_argument(
        "--overlap-lines",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Mark chunks from a dirty Git checkout training-eligible. "
            "Default is to index them for inspection but keep them ineligible."
        ),
    )

    parser.add_argument(
        "--allow-unversioned",
        action="store_true",
        help=(
            "Mark a non-Git content snapshot training-eligible. "
            "Default is inspection-only."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    manifest = build_code_corpus(
        root=args.root,
        logical_repository=args.repository,
        source_kind=args.source_kind,
        output_root=args.output_root,
        max_chars=args.max_chars,
        overlap_lines=args.overlap_lines,
        allow_dirty=args.allow_dirty,
        allow_unversioned=args.allow_unversioned,
    )

    if args.json:
        print(
            json.dumps(
                manifest.model_dump(
                    mode="json",
                    by_alias=True,
                ),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0

    print("Semantic Code Corpus")
    print("====================")
    print(f"Corpus:       {manifest.corpus_id}")
    print(
        f"Source:       "
        f"{manifest.source.logical_repository} "
        f"({manifest.source.source_kind})"
    )
    print(
        f"Revision:     "
        f"{manifest.source.revision_kind}:"
        f"{manifest.source.revision_id}"
    )
    print(
        f"Git dirty:    "
        f"{manifest.source.git_dirty}"
    )
    print(f"Chunks:       {manifest.chunk_count}")
    print(
        f"Eligible:     "
        f"{manifest.training_eligible_chunk_count}"
    )
    print(
        "Languages:    "
        + json.dumps(
            manifest.language_counts,
            sort_keys=True,
        )
    )
    print(
        "Audiences:    "
        + json.dumps(
            manifest.audience_counts,
            sort_keys=True,
        )
    )
    print(f"SHA-256:      {manifest.content_sha256}")
    print(
        "Training:     NOT PERFORMED"
    )
    print(
        "Promotion:    NOT PERFORMED"
    )

    if (
        manifest.source.git_dirty
        and manifest.training_eligible_chunk_count == 0
    ):
        print(
            "Note:         dirty Git checkout indexed for inspection only; "
            "commit/stabilize the source before training."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
