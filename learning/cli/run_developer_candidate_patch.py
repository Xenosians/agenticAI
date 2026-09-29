from __future__ import annotations

import argparse
import json

from pathlib import Path

from learning.cli.crashlog import (
    run_with_crashlog,
)

from learning.evaluation.developer_candidate_patch import (
    generate_developer_candidate_patch,
    prepare_developer_candidate_context,
)


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Prepare candidate-visible SWE repository context "
            "or generate a patch from an unpromoted developer "
            "checkpoint. Hidden evaluation metadata is never "
            "included in the model-visible context."
        )
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    prepare = (
        subparsers.add_parser(
            "prepare"
        )
    )

    prepare.add_argument(
        "--behavioral-holdout-dir",
        type=Path,
        required=True,
    )

    prepare.add_argument(
        "--case-id",
        required=True,
    )

    prepare.add_argument(
        "--allow-image-pull",
        action="store_true",
    )

    prepare.add_argument(
        "--context-chars",
        type=int,
        default=8000,
    )

    prepare.add_argument(
        "--json",
        action="store_true",
    )

    generate = (
        subparsers.add_parser(
            "generate"
        )
    )

    generate.add_argument(
        "--context-dir",
        type=Path,
        required=True,
    )

    generate.add_argument(
        "--heldout-eval-dir",
        type=Path,
        required=True,
    )

    generate.add_argument(
        "--checkpoint-id",
        required=True,
    )

    generate.add_argument(
        "--max-input-tokens",
        type=int,
        default=4096,
    )

    generate.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
    )

    generate.add_argument(
        "--json",
        action="store_true",
    )

    return parser


def _prepare(
    args,
) -> int:

    result = (
        prepare_developer_candidate_context(
            behavioral_holdout_directory=(
                args.behavioral_holdout_dir
            ),

            case_id=(
                args.case_id
            ),

            allow_image_pull=(
                args.allow_image_pull
            ),

            context_chars=(
                args.context_chars
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
        "Developer Candidate Context"
    )

    print(
        "==========================="
    )

    print(
        "Case         "
        + str(
            manifest.source_record_id
            or manifest.record_id
        )
    )

    print(
        "Parent       "
        + manifest.parent_holdout_id
    )

    print(
        "Base commit  VERIFIED"
    )

    print(
        "Tree         "
        + str(
            manifest.tree_entry_count
        )
        + " entries"
    )

    print(
        "Files        "
        + str(
            manifest.selected_file_count
        )
    )

    print(
        "Budget       "
        + str(
            manifest.context_character_budget
        )
        + " chars"
    )

    print(
        "Hidden eval  EXCLUDED"
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
        "Next gate    unpromoted checkpoint generation"
    )

    return 0


def _generate(
    args,
) -> int:

    result = (
        generate_developer_candidate_patch(
            context_directory=(
                args.context_dir
            ),

            heldout_evaluation_directory=(
                args.heldout_eval_dir
            ),

            checkpoint_id=(
                args.checkpoint_id
            ),

            max_input_tokens=(
                args.max_input_tokens
            ),

            max_new_tokens=(
                args.max_new_tokens
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

    report = (
        result.report
    )

    print()

    print(
        "Developer Candidate Patch"
    )

    print(
        "========================="
    )

    print(
        "Case         "
        + str(
            report.source_record_id
            or report.record_id
        )
    )

    print(
        "Checkpoint   "
        + report.checkpoint_id
    )

    print(
        "Model        "
        + report.target_model_key
    )

    print(
        "Backend      "
        + report.model_backend
    )

    print(
        "Lineage      "
        + (
            "VERIFIED"
            if report.heldout_lineage_verified
            else "FAILED"
        )
    )

    print(
        "Overlap      "
        + str(
            report.heldout_training_overlap_count
        )
    )

    print(
        "Heldout loss "
        + (
            "IMPROVED"
            if report.heldout_loss_improved
            else "NOT IMPROVED"
        )
    )

    print(
        "Base frozen  "
        + (
            "YES"
            if report.base_model_unchanged
            else "NO"
        )
    )

    print(
        "Input        "
        + str(
            report.input_tokens
        )
        + " tokens"
    )

    print(
        "Generated    "
        + str(
            report.generated_tokens
        )
        + " tokens"
    )

    print(
        "Patch        "
        + str(
            report.candidate_patch_bytes
        )
        + " bytes"
    )

    if (
        report.patch_text_validation_complete
    ):

        print(
            "Patch syntax "
            + (
                "VALID"
                if report.patch_text_valid
                else "INVALID"
            )
        )

    else:

        print(
            "Patch syntax NOT CHECKED"
        )

    if (
        report.patch_text_validation_errors
    ):

        for error in (
            report.patch_text_validation_errors
        ):

            print(
                "  validation "
                + error
            )

    print(
        "Hidden eval  NOT PROVIDED"
    )

    print(
        "Activation   NOT PERFORMED"
    )

    print(
        "Promotion    BLOCKED"
    )

    print(
        "Artifact     "
        + report.output_directory
    )

    print(
        "Candidate    "
        + report.candidate_patch_path
    )

    print()

    if (
        report.patch_text_validation_complete
        and report.patch_text_valid
    ):

        print(
            "Next gate    isolated behavioral sandbox"
        )

    elif (
        report.patch_text_validation_complete
    ):

        print(
            "Next gate    candidate patch format quality"
        )

    else:

        print(
            "Next gate    candidate patch validation"
        )

    return 0


def main() -> int:

    args = (
        build_parser()
        .parse_args()
    )

    if (
        args.command
        == "prepare"
    ):

        return (
            _prepare(
                args
            )
        )

    if (
        args.command
        == "generate"
    ):

        return (
            _generate(
                args
            )
        )

    raise RuntimeError(
        "Unknown candidate patch command."
    )


if __name__ == "__main__":

    raise SystemExit(
        run_with_crashlog(
            "developer-candidate-patch",
            main,
        )
    )
