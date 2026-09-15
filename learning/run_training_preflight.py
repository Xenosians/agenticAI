from __future__ import annotations

import argparse
import json

from pathlib import (
    Path,
)

from learning.training_preflight import (
    DpoQloraPreflight,
    DpoQloraRecipe,
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


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "training"
)


def build_parser(
) -> argparse.ArgumentParser:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Verify Phase-4 DPO + QLoRA training "
                "readiness without loading model weights "
                "or starting training."
            )
        )
    )

    parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    parser.add_argument(
        "--compute-dtype",
        choices=[
            "bfloat16",
            "float16",
        ],
        default=(
            "bfloat16"
        ),
    )

    parser.add_argument(
        "--minimum-vram-gib",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--allow-cpu",
        action="store_true",
    )

    parser.add_argument(
        "--allow-no-bf16",
        action="store_true",
    )

    parser.add_argument(
        "--write-manifest",
        action="store_true",
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    return parser


def main(
) -> int:

    args = (
        build_parser()
        .parse_args()
    )

    recipe = (
        DpoQloraRecipe(
            base_model_path=(
                args.model_path
            ),

            output_root=(
                args.output_root
            ),

            compute_dtype=(
                args.compute_dtype
            ),

            require_cuda=(
                not args.allow_cpu
            ),

            require_bfloat16=(
                not args.allow_no_bf16
            ),

            minimum_cuda_memory_gib=(
                args.minimum_vram_gib
            ),
        )
    )

    preflight = (
        DpoQloraPreflight(
            split_dir=(
                args.split_dir
            ),

            recipe=(
                recipe
            ),
        )
    )

    report = (
        preflight.run()
    )

    if args.json:

        print(
            json.dumps(
                report.model_dump(
                    mode="json",
                    by_alias=True,
                ),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )

    else:

        print(
            "DPO + QLoRA Training Preflight"
        )

        print(
            "=============================="
        )

        print(
            f"Split:      "
            f"{report.split_id or '-'}"
        )

        print(
            f"Dataset:    "
            f"{report.source_dataset_version or '-'}"
        )

        print(
            f"Train:      "
            f"{report.train_record_count}"
        )

        print(
            f"Validation: "
            f"{report.validation_record_count}"
        )

        print(
            f"CUDA:       "
            f"{'yes' if report.cuda.available else 'no'}"
        )

        if report.cuda.device_name:

            print(
                f"GPU:        "
                f"{report.cuda.device_name}"
            )

            print(
                f"VRAM:       "
                f"{report.cuda.total_memory_gib:.2f} GiB"
            )

        print()

        for check in report.checks:

            prefix = (
                "OK"
                if check.passed
                else "FAIL"
            )

            print(
                f"[{prefix}] "
                f"{check.name}: "
                f"{check.detail}"
            )

        print()

        print(
            "Training readiness: "
            + (
                "PASS"
                if report.ready
                else "FAIL"
            )
        )

    if (
        args.write_manifest
        and report.ready
    ):

        try:

            target = (
                preflight
                .write_dry_run_manifest(
                    report
                )
            )

        except Exception as exc:

            print(
                "ERROR: "
                f"{exc}"
            )

            return 1

        if not args.json:

            print(
                "Dry-run manifest: "
                f"{target}"
            )

    return (
        0
        if report.ready
        else 2
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )