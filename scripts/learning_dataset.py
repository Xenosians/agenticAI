from __future__ import annotations

import argparse
import json
import sys

from pathlib import (
    Path,
)


PROJECT_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        1
    ]
)

if (
    str(
        PROJECT_ROOT
    )
    not in sys.path
):
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from learning.datasets.dataset_loader import (  # noqa: E402
    PreferenceDatasetLoader,
)

from learning.datasets.dataset_verifier import (  # noqa: E402
    PreferenceDatasetVerifier,
)

from learning.datasets.records import (  # noqa: E402
    PreferenceDatasetBuilder,
)

from learning.evidence.types import (  # noqa: E402
    PreferenceExample,
)


DEFAULT_ROOT = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "datasets"
)


def load_examples(
    path: Path,
) -> list[
    PreferenceExample
]:

    resolved = (
        path
        .expanduser()
        .resolve()
    )

    if not resolved.is_file():
        raise ValueError(
            "Preference example input "
            f"does not exist: {resolved}"
        )

    examples: list[
        PreferenceExample
    ] = []

    for (
        line_number,
        line,
    ) in enumerate(
        resolved
        .read_text(
            encoding="utf-8"
        )
        .splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            raw = (
                json.loads(
                    line
                )
            )

        except Exception as exc:
            raise ValueError(
                (
                    "Invalid JSON at "
                    f"line {line_number}: "
                    f"{exc}"
                )
            ) from exc

        try:
            example = (
                PreferenceExample
                .model_validate(
                    raw
                )
            )

        except Exception as exc:
            raise ValueError(
                (
                    "Invalid preference "
                    f"example at line "
                    f"{line_number}: {exc}"
                )
            ) from exc

        examples.append(
            example
        )

    if not examples:
        raise ValueError(
            "No preference examples "
            "were found."
        )

    return examples


def command_list(
    root: Path,
) -> int:

    dataset_root = (
        root
        / "preference"
    )

    if not dataset_root.exists():
        print(
            "No preference datasets."
        )

        return 0

    versions = sorted(
        child.name

        for child
        in dataset_root.iterdir()

        if (
            child.is_dir()
            and child.name.startswith(
                "v"
            )
        )
    )

    if not versions:
        print(
            "No preference datasets."
        )

        return 0

    verifier = (
        PreferenceDatasetVerifier(
            root=root
        )
    )

    for version in versions:
        result = (
            verifier.verify(
                version
            )
        )

        state = (
            "OK"
            if result.ok
            else "INVALID"
        )

        print(
            f"{version} "
            f"{state} "
            f"records={result.record_count} "
            f"sha256="
            f"{result.content_sha256 or '-'}"
        )

    return 0


def command_verify(
    root: Path,
    version: str,
) -> int:

    verifier = (
        PreferenceDatasetVerifier(
            root=root
        )
    )

    result = (
        verifier.verify(
            version
        )
    )

    if result.ok:
        print(
            "Dataset verification passed."
        )

        print(
            f"Version: {result.version}"
        )

        print(
            f"Records: {result.record_count}"
        )

        print(
            "SHA-256: "
            f"{result.content_sha256}"
        )

        return 0

    print(
        "Dataset verification failed."
    )

    for error in (
        result.errors
    ):
        print(
            f"- {error}"
        )

    return 1


def command_show(
    root: Path,
    version: str,
) -> int:

    loader = (
        PreferenceDatasetLoader(
            root=root
        )
    )

    manifest, records = (
        loader.load(
            version
        )
    )

    print(
        manifest.model_dump_json(
            by_alias=True,
            indent=2,
        )
    )

    print(
        "\nRecords:"
    )

    for record in records:
        print(
            record.model_dump_json(
                by_alias=True,
                indent=2,
            )
        )

    return 0


def command_promote(
    root: Path,
    input_path: Path,
    promoted_by: str,
    reason: str,
) -> int:

    examples = (
        load_examples(
            input_path
        )
    )

    builder = (
        PreferenceDatasetBuilder(
            root=root
        )
    )

    manifest = (
        builder.promote(
            examples=(
                examples
            ),

            promoted_by=(
                promoted_by
            ),

            promotion_reason=(
                reason
            ),
        )
    )

    print(
        "Dataset promoted."
    )

    print(
        manifest.model_dump_json(
            by_alias=True,
            indent=2,
        )
    )

    return 0


def build_parser(
) -> argparse.ArgumentParser:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Learning dataset "
                "management CLI."
            )
        )
    )

    parser.add_argument(
        "--root",

        type=Path,

        default=(
            DEFAULT_ROOT
        ),

        help=(
            "Learning dataset root."
        ),
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    subparsers.add_parser(
        "list",
        help=(
            "List dataset versions."
        ),
    )

    verify = (
        subparsers.add_parser(
            "verify",

            help=(
                "Verify one dataset "
                "version."
            ),
        )
    )

    verify.add_argument(
        "version"
    )

    show = (
        subparsers.add_parser(
            "show",

            help=(
                "Load and display one "
                "verified version."
            ),
        )
    )

    show.add_argument(
        "version"
    )

    promote = (
        subparsers.add_parser(
            "promote",

            help=(
                "Promote canonical "
                "preference examples."
            ),
        )
    )

    promote.add_argument(
        "--input",

        type=Path,

        required=True,
    )

    promote.add_argument(
        "--promoted-by",

        choices=[
            "trusted_review",
            "evaluation",
        ],

        required=True,
    )

    promote.add_argument(
        "--reason",

        required=True,
    )

    return parser


def main(
) -> int:

    parser = (
        build_parser()
    )

    args = (
        parser.parse_args()
    )

    root = (
        args.root
        .expanduser()
        .resolve()
    )

    try:
        if (
            args.command
            == "list"
        ):
            return (
                command_list(
                    root
                )
            )

        if (
            args.command
            == "verify"
        ):
            return (
                command_verify(
                    root,
                    args.version,
                )
            )

        if (
            args.command
            == "show"
        ):
            return (
                command_show(
                    root,
                    args.version,
                )
            )

        if (
            args.command
            == "promote"
        ):
            return (
                command_promote(
                    root,
                    args.input,
                    args.promoted_by,
                    args.reason,
                )
            )

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        return 1

    parser.error(
        "Unknown command."
    )

    return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )