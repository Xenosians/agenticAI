from __future__ import annotations

import json

from pathlib import (
    Path,
)

from learning.dataset_verifier import (
    PreferenceDatasetVerifier,
)

from learning.types import (
    DatasetManifest,
    PreferenceDatasetRecord,
)


class PreferenceDatasetLoader:
    """
    Load only verified preference dataset versions.

    Corrupt or modified datasets are rejected before records are
    returned to training/evaluation code.
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

        self.verifier = (
            PreferenceDatasetVerifier(
                root=(
                    self.root
                )
            )
        )

    def load(
        self,
        version: str,
    ) -> tuple[
        DatasetManifest,
        list[
            PreferenceDatasetRecord
        ],
    ]:

        verification = (
            self.verifier
            .verify(
                version
            )
        )

        if not verification.ok:
            details = (
                "; ".join(
                    verification.errors
                )
            )

            raise ValueError(
                "Dataset verification failed: "
                f"{details}"
            )

        version_root = (
            self.root
            / "preference"
            / version
        )

        manifest_raw = (
            json.loads(
                (
                    version_root
                    / "manifest.json"
                )
                .read_text(
                    encoding="utf-8"
                )
            )
        )

        manifest = (
            DatasetManifest
            .model_validate(
                manifest_raw
            )
        )

        records: list[
            PreferenceDatasetRecord
        ] = []

        for line in (
            (
                version_root
                / "records.jsonl"
            )
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        ):
            if not line.strip():
                continue

            records.append(
                PreferenceDatasetRecord
                .model_validate(
                    json.loads(
                        line
                    )
                )
            )

        return (
            manifest,
            records,
        )