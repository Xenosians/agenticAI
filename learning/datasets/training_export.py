from __future__ import annotations

import hashlib
import json
import shutil

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

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from learning.curation.corpus_analysis import (
    load_eval_request_index,
    normalize_request,
)

from learning.datasets.dataset_loader import (
    PreferenceDatasetLoader,
)

from learning.evidence.sanitizer import (
    sanitize_value,
)

from learning.evidence.types import (
    PreferenceDatasetRecord,
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


DEFAULT_SPLIT_SEED = (
    "agentic-training-split.v1"
)


# ============================================================
# MANIFEST
# ============================================================


class TrainingSplitManifest(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "training-split-manifest.v1"
        ),
        alias="schema",
    )

    split_id: str

    created_at: str

    source_dataset_id: str

    source_dataset_version: str

    source_content_sha256: str

    split_seed: str

    requested_validation_fraction: float

    actual_validation_fraction: float

    record_count: int

    train_record_count: int

    validation_record_count: int

    unique_request_group_count: int

    held_out_request_count: int

    train_sha256: str

    validation_sha256: str

    train_record_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    validation_record_ids: list[
        str
    ] = Field(
        default_factory=list
    )


# ============================================================
# HELPERS
# ============================================================


def _utc_now(
) -> str:

    return (
        datetime
        .now(
            timezone.utc
        )
        .isoformat()
    )


def _canonical_json(
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


def _hash_bytes(
    value: bytes,
) -> str:

    return (
        hashlib
        .sha256(
            value
        )
        .hexdigest()
    )


def _group_sort_key(
    *,
    normalized_request: str,
    split_seed: str,
) -> str:

    value = (
        f"{split_seed}\0"
        f"{normalized_request}"
    )

    return (
        hashlib
        .sha256(
            value.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


def _serialize_records(
    records: list[
        PreferenceDatasetRecord
    ],
) -> str:

    serialized = [
        _canonical_json(
            sanitize_value(
                record.model_dump(
                    mode="json",
                    by_alias=True,
                )
            )
        )

        for record
        in records
    ]

    if not serialized:

        return ""

    return (
        "\n".join(
            serialized
        )
        + "\n"
    )


# ============================================================
# EXPORTER
# ============================================================


class PreferenceTrainingSplitExporter:
    """
    Export a verified immutable preference dataset into
    deterministic train/validation partitions.

    Guarantees:

    - source dataset must pass existing verification
    - held-out evaluation requests are rejected again
    - equal normalized user requests cannot cross splits
    - train and validation are both non-empty
    - split membership is deterministic for the same:
          dataset
          seed
          validation fraction
    - exported files are hashed in a manifest
    - existing exports are never overwritten
    """

    def __init__(
        self,
        *,
        dataset_root: Path,
        output_root: Path,
    ) -> None:

        self.dataset_root = (
            dataset_root
            .expanduser()
            .resolve()
        )

        self.output_root = (
            output_root
            .expanduser()
            .resolve()
        )

        self.loader = (
            PreferenceDatasetLoader(
                root=(
                    self.dataset_root
                )
            )
        )

    # ========================================================
    # HELD-OUT GUARD
    # ========================================================

    def _assert_no_held_out_intersection(
        self,
        *,
        records: list[
            PreferenceDatasetRecord
        ],
        eval_paths: list[
            Path
        ],
    ) -> int:

        if not eval_paths:

            return 0

        eval_index = (
            load_eval_request_index(
                eval_paths
            )
        )

        matches: list[
            str
        ] = []

        for record in records:

            normalized = (
                normalize_request(
                    record.user_request
                )
            )

            cases = (
                eval_index.get(
                    normalized
                )
            )

            if not cases:

                continue

            matches.append(
                (
                    f"record={record.record_id} "
                    f"source_example="
                    f"{record.source_example_id} "
                    f"eval_cases="
                    f"{','.join(cases)}"
                )
            )

        if matches:

            raise ValueError(
                "Held-out evaluation intersection "
                "detected during training split export: "
                + "; ".join(
                    matches
                )
            )

        return (
            len(
                eval_index
            )
        )

    # ========================================================
    # REQUEST GROUPING
    # ========================================================

    def _build_request_groups(
        self,
        records: list[
            PreferenceDatasetRecord
        ],
    ) -> dict[
        str,
        list[
            PreferenceDatasetRecord
        ],
    ]:

        groups: dict[
            str,
            list[
                PreferenceDatasetRecord
            ],
        ] = {}

        for record in records:

            normalized = (
                normalize_request(
                    record.user_request
                )
            )

            groups.setdefault(
                normalized,
                [],
            ).append(
                record
            )

        return groups

    # ========================================================
    # SPLIT
    # ========================================================

    def _split_records(
        self,
        *,
        records: list[
            PreferenceDatasetRecord
        ],
        validation_fraction: float,
        split_seed: str,
    ) -> tuple[
        list[
            PreferenceDatasetRecord
        ],
        list[
            PreferenceDatasetRecord
        ],
        int,
    ]:

        if (
            validation_fraction <= 0.0
            or validation_fraction >= 1.0
        ):

            raise ValueError(
                "validation_fraction must be "
                "greater than 0.0 and less than 1.0."
            )

        normalized_seed = (
            split_seed
            .strip()
        )

        if not normalized_seed:

            raise ValueError(
                "split_seed must not be empty."
            )

        if len(
            records
        ) < 2:

            raise ValueError(
                "At least two dataset records are "
                "required for train/validation export."
            )

        groups = (
            self._build_request_groups(
                records
            )
        )

        if len(
            groups
        ) < 2:

            raise ValueError(
                "At least two unique normalized "
                "request groups are required for "
                "train/validation export."
            )

        ordered_groups = (
            sorted(
                groups.items(),

                key=lambda item: (
                    _group_sort_key(
                        normalized_request=(
                            item[
                                0
                            ]
                        ),

                        split_seed=(
                            normalized_seed
                        ),
                    )
                ),
            )
        )

        target_validation_count = (
            round(
                len(
                    records
                )
                * validation_fraction
            )
        )

        target_validation_count = (
            max(
                1,
                min(
                    len(
                        records
                    )
                    - 1,
                    target_validation_count,
                ),
            )
        )

        validation_groups: list[
            tuple[
                str,
                list[
                    PreferenceDatasetRecord
                ],
            ]
        ] = []

        train_groups: list[
            tuple[
                str,
                list[
                    PreferenceDatasetRecord
                ],
            ]
        ] = []

        validation_count = 0

        for (
            index,
            group,
        ) in enumerate(
            ordered_groups
        ):

            groups_remaining = (
                len(
                    ordered_groups
                )
                - index
            )

            # Always preserve at least one complete request group
            # for training.
            if (
                validation_count
                < target_validation_count
                and groups_remaining
                > 1
            ):

                validation_groups.append(
                    group
                )

                validation_count += (
                    len(
                        group[
                            1
                        ]
                    )
                )

            else:

                train_groups.append(
                    group
                )

        train_records = [
            record

            for (
                _,
                group_records,
            )
            in train_groups

            for record
            in group_records
        ]

        validation_records = [
            record

            for (
                _,
                group_records,
            )
            in validation_groups

            for record
            in group_records
        ]

        if (
            not train_records
            or not validation_records
        ):

            raise ValueError(
                "Deterministic split produced an "
                "empty train or validation partition."
            )

        train_requests = {
            normalize_request(
                record.user_request
            )

            for record
            in train_records
        }

        validation_requests = {
            normalize_request(
                record.user_request
            )

            for record
            in validation_records
        }

        intersection = (
            train_requests
            & validation_requests
        )

        if intersection:

            raise ValueError(
                "Normalized request leakage detected "
                "between train and validation splits."
            )

        return (
            train_records,
            validation_records,
            len(
                groups
            ),
        )

    # ========================================================
    # SPLIT ID
    # ========================================================

    def _split_id(
        self,
        *,
        source_content_sha256: str,
        validation_fraction: float,
        split_seed: str,
    ) -> str:

        payload = {
            "source_content_sha256":
                source_content_sha256,

            "validation_fraction":
                validation_fraction,

            "split_seed":
                split_seed,
        }

        digest = (
            hashlib
            .sha256(
                _canonical_json(
                    payload
                )
                .encode(
                    "utf-8"
                )
            )
            .hexdigest()
        )

        return (
            "split-"
            f"{digest[:16]}"
        )

    # ========================================================
    # EXPORT
    # ========================================================

    def export(
        self,
        *,
        version: str,
        validation_fraction: float = 0.20,
        split_seed: str = (
            DEFAULT_SPLIT_SEED
        ),
        eval_paths: (
            Iterable[
                Path
            ]
            | None
        ) = None,
    ) -> TrainingSplitManifest:

        normalized_seed = (
            split_seed
            .strip()
        )

        if not normalized_seed:

            raise ValueError(
                "split_seed must not be empty."
            )

        (
            source_manifest,
            records,
        ) = (
            self.loader
            .load(
                version
            )
        )

        if eval_paths is None:

            resolved_eval_paths = (
                sorted(
                    DEFAULT_EVAL_DIRECTORY
                    .glob(
                        "*.jsonl"
                    )
                )
            )

        else:

            resolved_eval_paths = [
                Path(
                    path
                )
                .expanduser()
                .resolve()

                for path
                in eval_paths
            ]

        held_out_request_count = (
            self._assert_no_held_out_intersection(
                records=(
                    records
                ),

                eval_paths=(
                    resolved_eval_paths
                ),
            )
        )

        (
            train_records,
            validation_records,
            request_group_count,
        ) = (
            self._split_records(
                records=(
                    records
                ),

                validation_fraction=(
                    validation_fraction
                ),

                split_seed=(
                    normalized_seed
                ),
            )
        )

        train_blob = (
            _serialize_records(
                train_records
            )
        )

        validation_blob = (
            _serialize_records(
                validation_records
            )
        )

        train_sha256 = (
            _hash_bytes(
                train_blob.encode(
                    "utf-8"
                )
            )
        )

        validation_sha256 = (
            _hash_bytes(
                validation_blob.encode(
                    "utf-8"
                )
            )
        )

        split_id = (
            self._split_id(
                source_content_sha256=(
                    source_manifest
                    .content_sha256
                ),

                validation_fraction=(
                    validation_fraction
                ),

                split_seed=(
                    normalized_seed
                ),
            )
        )

        target = (
            self.output_root
            / source_manifest.dataset_id
            / source_manifest.version
            / split_id
        )

        if target.exists():

            raise ValueError(
                "Training split export already exists: "
                f"{target}"
            )

        target.mkdir(
            parents=True,
            exist_ok=False,
        )

        record_count = (
            len(
                records
            )
        )

        manifest = (
            TrainingSplitManifest(
                split_id=(
                    split_id
                ),

                created_at=(
                    _utc_now()
                ),

                source_dataset_id=(
                    source_manifest
                    .dataset_id
                ),

                source_dataset_version=(
                    source_manifest
                    .version
                ),

                source_content_sha256=(
                    source_manifest
                    .content_sha256
                ),

                split_seed=(
                    normalized_seed
                ),

                requested_validation_fraction=(
                    validation_fraction
                ),

                actual_validation_fraction=(
                    len(
                        validation_records
                    )
                    / record_count
                ),

                record_count=(
                    record_count
                ),

                train_record_count=(
                    len(
                        train_records
                    )
                ),

                validation_record_count=(
                    len(
                        validation_records
                    )
                ),

                unique_request_group_count=(
                    request_group_count
                ),

                held_out_request_count=(
                    held_out_request_count
                ),

                train_sha256=(
                    train_sha256
                ),

                validation_sha256=(
                    validation_sha256
                ),

                train_record_ids=[
                    record.record_id

                    for record
                    in train_records
                ],

                validation_record_ids=[
                    record.record_id

                    for record
                    in validation_records
                ],
            )
        )

        try:

            (
                target
                / "train.jsonl"
            ).write_text(
                train_blob,
                encoding="utf-8",
            )

            (
                target
                / "validation.jsonl"
            ).write_text(
                validation_blob,
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

            (
                target
                / "manifest.json"
            ).write_text(
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