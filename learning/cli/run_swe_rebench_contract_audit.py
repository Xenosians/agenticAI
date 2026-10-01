from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from learning.continual.corpus_materializer import (
    _iter_source,
    normalize_source_row,
)
from learning.continual.corpus_registry import (
    load_corpus_registry,
)
from learning.paths import (
    REPOSITORY_ROOT,
)


DEFAULT_CORPUS_REGISTRY = (
    REPOSITORY_ROOT
    / "config"
    / "continual_corpus_registry.json"
)
from learning.developer_contract import (
    behavioral_parser_supported,
    canonical_language,
    extract_log_parser,
    parse_swe_rebench_raw_row,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the pinned SWE-rebench corpus against the local "
            "developer training/evaluation contract."
        )
    )

    parser.add_argument(
        "--registry",
        default=str(
            DEFAULT_CORPUS_REGISTRY
        ),
    )

    parser.add_argument(
        "--source-id",
        default="swe-rebench-v2",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=8000,
        help=(
            "Maximum upstream rows to scan. 8000 covers the current "
            "Filtered-Verified corpus while keeping the reader bounded."
        ),
    )

    return parser


def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )

    if args.max_rows <= 0:
        raise SystemExit(
            "--max-rows must be positive"
        )

    registry = load_corpus_registry(
        args.registry
    )

    source = next(
        (
            item
            for item
            in registry.sources
            if item.source_id
            == args.source_id
        ),
        None,
    )

    if source is None:
        raise SystemExit(
            "Unknown corpus source: "
            + args.source_id
        )

    languages = Counter()
    parsers = Counter()
    missing_fields = Counter()
    normalization = Counter()
    parser_capability = Counter()

    scanned = 0

    iterator = _iter_source(
        source,
        start_index=0,
        max_rows=args.max_rows,
    )

    try:
        for index, row in iterator:
            scanned += 1

            for field_name in (
                "instance_id",
                "repo",
                "base_commit",
                "patch",
                "problem_statement",
                "install_config",
            ):
                value = row.get(
                    field_name
                )

                if value is None or value == "":
                    missing_fields[
                        field_name
                    ] += 1

            raw_language = row.get(
                "language"
            )

            language = canonical_language(
                raw_language
                if isinstance(
                    raw_language,
                    str,
                )
                else None
            )

            languages[
                language or "<missing>"
            ] += 1

            parser_name = extract_log_parser(
                row
            )

            parsers[
                parser_name or "<missing>"
            ] += 1

            parser_capability[
                (
                    "supported"
                    if behavioral_parser_supported(
                        parser_name
                    )
                    else "unsupported_or_missing"
                )
            ] += 1

            try:
                parse_swe_rebench_raw_row(
                    row
                )
            except Exception:
                normalization[
                    "raw_contract_invalid"
                ] += 1
                continue

            try:
                record = normalize_source_row(
                    source,
                    index=index,
                    row=row,
                )
            except Exception:
                normalization[
                    "normalization_error"
                ] += 1
                continue

            if record is None:
                normalization[
                    "normalization_none"
                ] += 1
            else:
                normalization[
                    "normalization_ok"
                ] += 1

    finally:
        close = getattr(
            iterator,
            "close",
            None,
        )

        if callable(
            close
        ):
            close()

    report: dict[str, Any] = {
        "schema":
            "swe-rebench-contract-audit.v1",

        "source_id":
            source.source_id,

        "dataset_id":
            source.dataset_id,

        "revision":
            source.revision,

        "scanned":
            scanned,

        "canonical_languages":
            dict(
                languages.most_common()
            ),

        "behavioral_log_parsers":
            dict(
                parsers.most_common()
            ),

        "behavioral_parser_capability":
            dict(
                parser_capability
            ),

        "missing_fields":
            dict(
                missing_fields
            ),

        "normalization":
            dict(
                normalization
            ),
    }

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
