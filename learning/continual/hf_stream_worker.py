from __future__ import annotations

import argparse
import json
import os
import sys

from typing import Any


def _json_default(
    value: Any,
):
    if isinstance(
        value,
        bytes,
    ):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    return str(
        value
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Disposable isolated Hugging Face streaming reader."
        )
    )

    parser.add_argument(
        "--dataset-id",
        required=True,
    )

    parser.add_argument(
        "--subset",
        default=None,
    )

    parser.add_argument(
        "--split",
        default="train",
    )

    parser.add_argument(
        "--revision",
        default=None,
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        required=True,
    )

    return parser


def _exit(
    code: int,
) -> None:
    """
    This process is deliberately disposable.

    Do NOT allow normal CPython interpreter teardown here.

    Some Hugging Face streaming / PyArrow combinations can crash while
    native Python-backed objects are being finalized after a bounded
    stream. os._exit() terminates this non-authoritative reader without
    running those finalizers.

    All authoritative state lives in the parent process.
    """

    try:
        sys.stdout.flush()
    except Exception:
        pass

    try:
        sys.stderr.flush()
    except Exception:
        pass

    os._exit(
        code
    )


def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )

    if args.max_rows <= 0:
        print(
            "[HF-CORPUS-WORKER] --max-rows must be positive",
            file=sys.stderr,
            flush=True,
        )

        _exit(
            2
        )

    os.environ.setdefault(
        "HF_DATASETS_DISABLE_PROGRESS_BARS",
        "1",
    )

    try:
        from datasets import (
            load_dataset,
        )

        dataset = load_dataset(
            args.dataset_id,
            name=(
                args.subset
                or None
            ),
            split=args.split,
            revision=(
                args.revision
                or None
            ),
            streaming=True,
        )

        start_index = max(
            0,
            args.start_index,
        )

        emitted = 0

        if (
            start_index > 0
            and hasattr(
                dataset,
                "skip",
            )
        ):
            dataset = dataset.skip(
                start_index
            )

            base_index = (
                start_index
            )

        else:
            base_index = 0

        for offset, row in enumerate(
            dataset
        ):
            absolute_index = (
                base_index
                + offset
            )

            if (
                base_index == 0
                and absolute_index
                < start_index
            ):
                continue

            if not isinstance(
                row,
                dict,
            ):
                continue

            envelope = {
                "index":
                    absolute_index,

                "row":
                    row,
            }

            sys.stdout.write(
                json.dumps(
                    envelope,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                    default=(
                        _json_default
                    ),
                )
                + "\n"
            )

            sys.stdout.flush()

            emitted += 1

            if (
                emitted
                >= args.max_rows
            ):
                # Critical:
                #
                # Exit while the dataset/native objects are still alive.
                # Do not return from this function and let CPython tear
                # them down.
                _exit(
                    0
                )

        # Dataset exhausted before max_rows.
        #
        # Still bypass native interpreter teardown.
        _exit(
            0
        )

    except BrokenPipeError:
        # Parent already collected its bounded window.
        _exit(
            0
        )

    except BaseException as exc:
        try:
            print(
                "[HF-CORPUS-WORKER] "
                + repr(
                    exc
                ),
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass

        _exit(
            1
        )


if __name__ == "__main__":
    main()
