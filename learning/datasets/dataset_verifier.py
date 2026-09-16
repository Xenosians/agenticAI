from __future__ import annotations

import hashlib
import json

from dataclasses import (
    dataclass,
    field,
)

from pathlib import (
    Path,
)

from pydantic import (
    ValidationError,
)

from learning.evidence.types import (
    DatasetManifest,
    PreferenceDatasetRecord,
)


@dataclass
class DatasetVerificationResult:
    ok: bool

    version: (
        str | None
    ) = None

    record_count: int = 0

    content_sha256: (
        str | None
    ) = None

    errors: list[
        str
    ] = field(
        default_factory=list
    )


class PreferenceDatasetVerifier:
    """
    Verify one immutable preference dataset version.

    Verification covers:

    - expected files
    - manifest schema
    - dataset type
    - version/path agreement
    - content SHA-256
    - record count
    - record schema
    - dataset eligibility
    - duplicate record IDs
    - duplicate source example IDs
    - manifest/source agreement
    """

    def __init__(
        self,
        *,
        root: Path,
    ) -> None:

        self.root = (
            root
            .expanduser()
            .resolve()
        )

    def _version_root(
        self,
        version: str,
    ) -> Path:

        normalized = (
            version
            .strip()
        )

        if not normalized:
            raise ValueError(
                "Dataset version must not "
                "be empty."
            )

        return (
            self.root
            / "preference"
            / normalized
        )

    def verify(
        self,
        version: str,
    ) -> DatasetVerificationResult:

        errors: list[str] = []

        version_root = (
            self._version_root(
                version
            )
        )

        if not version_root.exists():
            return (
                DatasetVerificationResult(
                    ok=False,

                    version=(
                        version
                    ),

                    errors=[
                        (
                            "Dataset version does "
                            "not exist."
                        )
                    ],
                )
            )

        if not version_root.is_dir():
            return (
                DatasetVerificationResult(
                    ok=False,

                    version=(
                        version
                    ),

                    errors=[
                        (
                            "Dataset version path "
                            "is not a directory."
                        )
                    ],
                )
            )

        manifest_path = (
            version_root
            / "manifest.json"
        )

        records_path = (
            version_root
            / "records.jsonl"
        )

        if not manifest_path.is_file():
            errors.append(
                "manifest.json is missing."
            )

        if not records_path.is_file():
            errors.append(
                "records.jsonl is missing."
            )

        if errors:
            return (
                DatasetVerificationResult(
                    ok=False,

                    version=(
                        version
                    ),

                    errors=(
                        errors
                    ),
                )
            )

        # ========================================================
        # MANIFEST
        # ========================================================

        try:
            manifest_raw = (
                json.loads(
                    manifest_path
                    .read_text(
                        encoding="utf-8"
                    )
                )
            )

        except Exception as exc:
            return (
                DatasetVerificationResult(
                    ok=False,

                    version=(
                        version
                    ),

                    errors=[
                        (
                            "manifest.json could "
                            "not be parsed: "
                            f"{exc}"
                        )
                    ],
                )
            )

        try:
            manifest = (
                DatasetManifest
                .model_validate(
                    manifest_raw
                )
            )

        except ValidationError as exc:
            return (
                DatasetVerificationResult(
                    ok=False,

                    version=(
                        version
                    ),

                    errors=[
                        (
                            "manifest.json failed "
                            "schema validation: "
                            f"{exc}"
                        )
                    ],
                )
            )

        if (
            manifest.schema_name
            != "dataset-manifest.v1"
        ):
            errors.append(
                "Unexpected manifest schema."
            )

        if (
            manifest.dataset_id
            != "preference"
        ):
            errors.append(
                "Unexpected dataset_id."
            )

        if (
            manifest.dataset_type
            != "preference"
        ):
            errors.append(
                "Unexpected dataset_type."
            )

        if (
            manifest.record_schema
            != (
                "preference-dataset-record.v1"
            )
        ):
            errors.append(
                "Unexpected record_schema."
            )

        if (
            manifest.version
            != version_root.name
        ):
            errors.append(
                "Manifest version does not "
                "match directory name."
            )

        # ========================================================
        # EXACT FILE HASH
        # ========================================================

        records_blob = (
            records_path
            .read_bytes()
        )

        calculated_sha256 = (
            hashlib
            .sha256(
                records_blob
            )
            .hexdigest()
        )

        if (
            calculated_sha256
            != manifest.content_sha256
        ):
            errors.append(
                "records.jsonl SHA-256 does "
                "not match manifest."
            )

        # ========================================================
        # RECORDS
        # ========================================================

        records: list[
            PreferenceDatasetRecord
        ] = []

        for (
            line_number,
            line,
        ) in enumerate(
            records_path
            .read_text(
                encoding="utf-8"
            )
            .splitlines(),
            start=1,
        ):
            if not line.strip():
                errors.append(
                    (
                        "Empty JSONL record at "
                        f"line {line_number}."
                    )
                )

                continue

            try:
                raw_record = (
                    json.loads(
                        line
                    )
                )

            except Exception as exc:
                errors.append(
                    (
                        "Invalid JSON at line "
                        f"{line_number}: {exc}"
                    )
                )

                continue

            try:
                record = (
                    PreferenceDatasetRecord
                    .model_validate(
                        raw_record
                    )
                )

            except ValidationError as exc:
                errors.append(
                    (
                        "Invalid dataset record "
                        f"at line {line_number}: "
                        f"{exc}"
                    )
                )

                continue

            if (
                record.schema_name
                != (
                    "preference-dataset-record.v1"
                )
            ):
                errors.append(
                    (
                        "Unexpected record schema "
                        f"at line {line_number}."
                    )
                )

            if (
                record.dataset_eligible
                is not True
            ):
                errors.append(
                    (
                        "Dataset record at line "
                        f"{line_number} is not "
                        "eligible."
                    )
                )

            if (
                record.rejected.model_dump()
                == record.chosen.model_dump()
            ):
                errors.append(
                    (
                        "Chosen and rejected "
                        "behavior are identical "
                        f"at line {line_number}."
                    )
                )

            records.append(
                record
            )

        # ========================================================
        # COUNT
        # ========================================================

        if (
            len(
                records
            )
            != manifest.record_count
        ):
            errors.append(
                (
                    "Manifest record_count "
                    "does not match parsed "
                    "records."
                )
            )

        # ========================================================
        # DUPLICATES
        # ========================================================

        record_ids = [
            record.record_id

            for record
            in records
        ]

        if (
            len(
                record_ids
            )
            != len(
                set(
                    record_ids
                )
            )
        ):
            errors.append(
                "Duplicate record_id detected."
            )

        source_example_ids = [
            record.source_example_id

            for record
            in records
        ]

        if (
            len(
                source_example_ids
            )
            != len(
                set(
                    source_example_ids
                )
            )
        ):
            errors.append(
                (
                    "Duplicate source_example_id "
                    "detected."
                )
            )

        if (
            source_example_ids
            != manifest.source_example_ids
        ):
            errors.append(
                (
                    "Manifest source_example_ids "
                    "do not match dataset records."
                )
            )

        return (
            DatasetVerificationResult(
                ok=(
                    not errors
                ),

                version=(
                    manifest.version
                ),

                record_count=(
                    len(
                        records
                    )
                ),

                content_sha256=(
                    calculated_sha256
                ),

                errors=(
                    errors
                ),
            )
        )