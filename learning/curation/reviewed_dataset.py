from __future__ import annotations

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
    load_corrections,
    load_trajectories,
)

from learning.curation.engine import (
    CurationReport,
    curate_corpus,
)

from learning.datasets.records import (
    PreferenceDatasetBuilder,
)

from learning.curation.preferences import (
    build_preference_example,
)

from learning.evidence.types import (
    DatasetManifest,
    PreferenceExample,
)


class ReviewedDatasetBuildResult(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "reviewed-dataset-build-result.v1"
        ),
        alias="schema",
    )

    curation: CurationReport

    preference_example_count: int

    source_trajectory_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    source_correction_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    manifest: DatasetManifest


class ReviewedPreferenceDatasetBuilder:
    """
    Trusted bridge:

        immutable raw evidence
            +
        trusted review ledger
            ->
        Phase 3B curation
            ->
        canonical preference examples
            ->
        immutable preference dataset

    Only corrections explicitly marked usable by curation can
    generate preference examples.
    """

    def __init__(
        self,
        *,
        trajectory_path: Path,
        correction_path: Path,
        review_path: Path,
        dataset_root: Path,
    ) -> None:

        self.trajectory_path = (
            trajectory_path
            .expanduser()
            .resolve()
        )

        self.correction_path = (
            correction_path
            .expanduser()
            .resolve()
        )

        self.review_path = (
            review_path
            .expanduser()
            .resolve()
        )

        self.dataset_root = (
            dataset_root
            .expanduser()
            .resolve()
        )

    def build(
        self,
        *,
        eval_paths: Iterable[
            Path
        ] = (),
        promoted_by: str = (
            "trusted_review"
        ),
        promotion_reason: str,
    ) -> ReviewedDatasetBuildResult:

        resolved_eval_paths = [
            Path(
                path
            )
            .expanduser()
            .resolve()

            for path
            in eval_paths
        ]

        curation_report = (
            curate_corpus(
                trajectory_path=(
                    self.trajectory_path
                ),

                correction_path=(
                    self.correction_path
                ),

                review_path=(
                    self.review_path
                ),

                eval_paths=(
                    resolved_eval_paths
                ),
            )
        )

        trajectories = (
            load_trajectories(
                self.trajectory_path
            )
        )

        corrections = (
            load_corrections(
                self.correction_path
            )
        )

        trajectory_by_id = {
            trajectory.trajectory_id:
                trajectory

            for trajectory
            in trajectories
        }

        correction_by_id = {
            correction.correction_id:
                correction

            for correction
            in corrections
        }

        # ====================================================
        # BUILD ONLY FROM CURATED ELIGIBLE EVIDENCE
        # ====================================================

        examples: list[
            PreferenceExample
        ] = []

        source_trajectory_ids: list[
            str
        ] = []

        source_correction_ids: list[
            str
        ] = []

        for curated in (
            curation_report.eligible
        ):

            trajectory = (
                trajectory_by_id.get(
                    curated.trajectory_id
                )
            )

            if trajectory is None:

                raise ValueError(
                    "Curation report references "
                    "missing trajectory: "
                    f"{curated.trajectory_id}"
                )

            for correction_id in (
                curated.usable_correction_ids
            ):

                correction = (
                    correction_by_id.get(
                        correction_id
                    )
                )

                if correction is None:

                    raise ValueError(
                        "Curation report references "
                        "missing correction: "
                        f"{correction_id}"
                    )

                if (
                    correction.trajectory_id
                    != trajectory.trajectory_id
                ):

                    raise ValueError(
                        "Correction lineage mismatch: "
                        f"{correction_id}"
                    )

                example = (
                    build_preference_example(
                        trajectory=(
                            trajectory
                        ),

                        correction=(
                            correction
                        ),
                    )
                )

                examples.append(
                    example
                )

                source_trajectory_ids.append(
                    trajectory.trajectory_id
                )

                source_correction_ids.append(
                    correction.correction_id
                )

        if not examples:

            raise ValueError(
                "No trusted reviewed preference examples "
                "are available for dataset promotion."
            )

        # ====================================================
        # DEFENSE IN DEPTH
        #
        # PreferenceDatasetBuilder independently performs the
        # held-out contamination guard again.
        # ====================================================

        builder = (
            PreferenceDatasetBuilder(
                root=(
                    self.dataset_root
                ),

                eval_paths=(
                    resolved_eval_paths
                ),
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
                    promotion_reason
                ),
            )
        )

        return (
            ReviewedDatasetBuildResult(
                curation=(
                    curation_report
                ),

                preference_example_count=(
                    len(
                        examples
                    )
                ),

                source_trajectory_ids=(
                    source_trajectory_ids
                ),

                source_correction_ids=(
                    source_correction_ids
                ),

                manifest=(
                    manifest
                ),
            )
        )