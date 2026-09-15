from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid

from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
)

from typing import (
    Iterable,
)

from learning.corpus_analysis import (
    load_eval_request_index,
    normalize_request,
)

from learning.sanitizer import (
    sanitize_value,
)

from learning.types import (
    DatasetManifest,
    DatasetPromotion,
    PreferenceDatasetRecord,
    PreferenceExample,
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


DEFAULT_EVAL_DIRECTORY = (
    PROJECT_ROOT
    / "learning"
    / "evals"
)


VERSION_PATTERN = re.compile(
    r"^v([0-9]{6})$"
)


VALID_PROMOTION_SOURCES = {
    "trusted_review",
    "evaluation",
}


class PreferenceDatasetBuilder:
    """
    Build immutable versioned preference datasets.

    Source preference examples are never mutated.

    Every dataset version contains:

        manifest.json
        records.jsonl

    Existing versions are never overwritten.

    Held-out evaluation contamination is checked again immediately
    before promotion as defense in depth.
    """

    def __init__(
        self,
        *,
        root: Path,
        eval_paths: (
            Iterable[
                Path
            ]
            | None
        ) = None,
    ) -> None:

        self.root = (
            root
            .expanduser()
            .resolve()
        )

        if eval_paths is None:

            self.eval_paths = (
                sorted(
                    DEFAULT_EVAL_DIRECTORY
                    .glob(
                        "*.jsonl"
                    )
                )
            )

        else:

            self.eval_paths = [
                Path(
                    path
                )
                .expanduser()
                .resolve()

                for path
                in eval_paths
            ]

    def _dataset_root(
        self,
    ) -> Path:

        return (
            self.root
            / "preference"
        )

    def _next_version(
        self,
    ) -> str:

        dataset_root = (
            self._dataset_root()
        )

        if not dataset_root.exists():

            return (
                "v000001"
            )

        highest = 0

        for child in (
            dataset_root
            .iterdir()
        ):

            if not child.is_dir():

                continue

            match = (
                VERSION_PATTERN
                .match(
                    child.name
                )
            )

            if match is None:

                continue

            value = (
                int(
                    match.group(
                        1
                    )
                )
            )

            highest = (
                max(
                    highest,
                    value,
                )
            )

        return (
            f"v{highest + 1:06d}"
        )

    def _canonical_json(
        self,
        value: dict,
    ) -> str:

        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
            )
        )

    # ========================================================
    # HELD-OUT PROMOTION GUARD
    # ========================================================

    def _assert_no_held_out_contamination(
        self,
        examples: list[
            PreferenceExample
        ],
    ) -> None:

        if not self.eval_paths:

            return

        eval_index = (
            load_eval_request_index(
                self.eval_paths
            )
        )

        matches: list[
            tuple[
                str,
                str,
                list[
                    str
                ],
            ]
        ] = []

        for example in examples:

            normalized_request = (
                normalize_request(
                    example.user_request
                )
            )

            eval_cases = (
                eval_index.get(
                    normalized_request
                )
            )

            if not eval_cases:

                continue

            matches.append(
                (
                    example.example_id,
                    example.trajectory_id,
                    list(
                        eval_cases
                    ),
                )
            )

        if not matches:

            return

        details = (
            "; ".join(
                (
                    f"example={example_id} "
                    f"trajectory={trajectory_id} "
                    f"eval_cases={','.join(eval_cases)}"
                )

                for (
                    example_id,
                    trajectory_id,
                    eval_cases,
                )
                in matches
            )
        )

        raise ValueError(
            "Held-out evaluation contamination "
            "detected during dataset promotion: "
            f"{details}"
        )

    # ========================================================
    # RECORD BUILDING
    # ========================================================

    def _build_record(
        self,
        *,
        example: PreferenceExample,
        promoted_by: str,
        promotion_reason: str,
        promoted_at: str,
    ) -> PreferenceDatasetRecord:

        if (
            example.rejected.model_dump()
            == example.chosen.model_dump()
        ):

            raise ValueError(
                "Preference example contains "
                "identical chosen and rejected "
                "behavior."
            )

        return (
            PreferenceDatasetRecord(
                record_id=(
                    uuid
                    .uuid4()
                    .hex
                ),

                source_example_id=(
                    example.example_id
                ),

                trajectory_id=(
                    example.trajectory_id
                ),

                correction_id=(
                    example.correction_id
                ),

                task_id=(
                    example.task_id
                ),

                source=(
                    example.source
                ),

                correction_type=(
                    example.correction_type
                ),

                user_request=(
                    example.user_request
                ),

                rejected=(
                    example
                    .rejected
                    .model_copy(
                        deep=True
                    )
                ),

                chosen=(
                    example
                    .chosen
                    .model_copy(
                        deep=True
                    )
                ),

                promotion=(
                    DatasetPromotion(
                        promoted_at=(
                            promoted_at
                        ),

                        promoted_by=(
                            promoted_by
                        ),

                        reason=(
                            promotion_reason
                        ),
                    )
                ),

                dataset_eligible=True,
            )
        )

    # ========================================================
    # PROMOTION
    # ========================================================

    def promote(
        self,
        *,
        examples: list[
            PreferenceExample
        ],
        promoted_by: str,
        promotion_reason: str,
    ) -> DatasetManifest:

        if not examples:

            raise ValueError(
                "At least one preference example "
                "is required."
            )

        normalized_promoted_by = (
            promoted_by
            .strip()
            .lower()
        )

        if (
            normalized_promoted_by
            not in VALID_PROMOTION_SOURCES
        ):

            raise ValueError(
                "Dataset promotion must come from "
                "trusted_review or evaluation."
            )

        normalized_reason = (
            promotion_reason
            .strip()
        )

        if not normalized_reason:

            raise ValueError(
                "promotion_reason must not be empty."
            )

        example_ids = [
            example.example_id

            for example
            in examples
        ]

        if (
            len(
                example_ids
            )
            != len(
                set(
                    example_ids
                )
            )
        ):

            raise ValueError(
                "Duplicate preference examples "
                "cannot appear in one dataset version."
            )

        # ----------------------------------------------------
        # Defense in depth.
        #
        # Curation should already have removed held-out
        # contamination, but promotion independently checks the
        # exact normalized user requests again.
        # ----------------------------------------------------

        self._assert_no_held_out_contamination(
            examples
        )

        created_at = (
            datetime
            .now(
                timezone.utc
            )
            .isoformat()
        )

        records = [
            self._build_record(
                example=(
                    example
                ),

                promoted_by=(
                    normalized_promoted_by
                ),

                promotion_reason=(
                    normalized_reason
                ),

                promoted_at=(
                    created_at
                ),
            )

            for example
            in examples
        ]

        sanitized_records = [
            sanitize_value(
                record.model_dump(
                    mode="json",
                    by_alias=True,
                )
            )

            for record
            in records
        ]

        serialized_records = [
            self._canonical_json(
                record
            )

            for record
            in sanitized_records
        ]

        records_blob = (
            "\n".join(
                serialized_records
            )
            + "\n"
        )

        content_sha256 = (
            hashlib
            .sha256(
                records_blob.encode(
                    "utf-8"
                )
            )
            .hexdigest()
        )

        dataset_root = (
            self._dataset_root()
        )

        dataset_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Reserve an immutable version directory.
        # ----------------------------------------------------

        while True:

            version = (
                self._next_version()
            )

            target = (
                dataset_root
                / version
            )

            try:

                target.mkdir(
                    parents=False,
                    exist_ok=False,
                )

                break

            except FileExistsError:

                continue

        manifest = (
            DatasetManifest(
                dataset_id=(
                    "preference"
                ),

                version=(
                    version
                ),

                created_at=(
                    created_at
                ),

                dataset_type=(
                    "preference"
                ),

                record_schema=(
                    "preference-dataset-record.v1"
                ),

                record_count=(
                    len(
                        sanitized_records
                    )
                ),

                content_sha256=(
                    content_sha256
                ),

                promoted_by=(
                    normalized_promoted_by
                ),

                promotion_reason=(
                    normalized_reason
                ),

                source_example_ids=(
                    example_ids
                ),
            )
        )

        try:

            records_path = (
                target
                / "records.jsonl"
            )

            records_path.write_text(
                records_blob,
                encoding="utf-8",
            )

            manifest_payload = (
                sanitize_value(
                    manifest.model_dump(
                        mode="json",
                        by_alias=True,
                    )
                )
            )

            manifest_path = (
                target
                / "manifest.json"
            )

            manifest_path.write_text(
                json.dumps(
                    manifest_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        except Exception:

            shutil.rmtree(
                target,
                ignore_errors=True,
            )

            raise

        return (
            manifest
        )