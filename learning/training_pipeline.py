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

from learning.corpus_analysis import (
    load_trajectories,
    normalize_request,
)

from learning.curation import (
    CurationReport,
    curate_corpus,
)

from learning.dataset_loader import (
    PreferenceDatasetLoader,
)

from learning.diversity_gate import (
    DiversityGatePolicy,
    DiversityGateReport,
    evaluate_diversity_gate,
)

from learning.training_export import (
    DEFAULT_EVAL_DIRECTORY,
    DEFAULT_SPLIT_SEED,
    PreferenceTrainingSplitExporter,
    TrainingSplitManifest,
)

from learning.types import (
    PreferenceDatasetRecord,
)


# ============================================================
# RESULT
# ============================================================


class TrustedTrainingPipelineResult(
    BaseModel
):
    model_config = (
        ConfigDict(
            populate_by_name=True
        )
    )

    schema_name: str = Field(
        default=(
            "trusted-training-pipeline-result.v1"
        ),
        alias="schema",
    )

    source_dataset_version: str

    dataset_record_count: int

    curated_eligible_count: int

    curation: CurationReport

    diversity: DiversityGateReport

    split: TrainingSplitManifest


# ============================================================
# PIPELINE
# ============================================================


class TrustedTrainingPipeline:
    """
    Trusted path from raw learning evidence to an exported
    train/validation split.

    The pipeline deliberately reuses every previous safety layer:

        raw evidence
            ->
        deterministic curation
            ->
        contamination rejection
            ->
        diversity / balance gate
            ->
        verified promoted dataset
            ->
        provenance validation
            ->
        deterministic training export

    No step makes earlier safety checks redundant.
    """

    def __init__(
        self,
        *,
        trajectory_path: Path,
        correction_path: Path,
        dataset_root: Path,
        output_root: Path,
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

        self.dataset_loader = (
            PreferenceDatasetLoader(
                root=(
                    self.dataset_root
                )
            )
        )

        self.exporter = (
            PreferenceTrainingSplitExporter(
                dataset_root=(
                    self.dataset_root
                ),

                output_root=(
                    self.output_root
                ),
            )
        )

    # ========================================================
    # EVALUATION PATHS
    # ========================================================

    def _resolve_eval_paths(
        self,
        eval_paths: (
            Iterable[
                Path
            ]
            | None
        ),
    ) -> list[
        Path
    ]:

        if eval_paths is None:

            return (
                sorted(
                    DEFAULT_EVAL_DIRECTORY
                    .glob(
                        "*.jsonl"
                    )
                )
            )

        return [
            Path(
                path
            )
            .expanduser()
            .resolve()

            for path
            in eval_paths
        ]

    # ========================================================
    # CURATION SAFETY
    # ========================================================

    def _assert_curation_safe(
        self,
        report: CurationReport,
    ) -> None:

        if (
            report
            .held_out_contamination_count
            <= 0
        ):

            return

        details: list[
            str
        ] = []

        for match in (
            report
            .contamination_matches
        ):

            details.append(
                (
                    f"{match.trajectory_id}:"
                    f"{','.join(match.eval_cases)}"
                )
            )

        suffix = (
            "; ".join(
                details
            )
        )

        if suffix:

            suffix = (
                " "
                + suffix
            )

        raise ValueError(
            "Curation report contains held-out "
            "evaluation contamination."
            f"{suffix}"
        )

    # ========================================================
    # DIVERSITY SAFETY
    # ========================================================

    def _assert_diversity_safe(
        self,
        report: DiversityGateReport,
    ) -> None:

        if (
            report
            .promotion_eligible
        ):

            return

        failed = (
            ", ".join(
                report.failed_checks
            )
        )

        raise ValueError(
            "Diversity/balance gate failed: "
            f"{failed}"
        )

    # ========================================================
    # DATASET LINEAGE
    # ========================================================

    def _assert_dataset_lineage(
        self,
        *,
        records: list[
            PreferenceDatasetRecord
        ],
        curation_report: CurationReport,
    ) -> None:
        """
        Verify that every training record comes from evidence that
        Phase 3B explicitly allowed.

        We verify:

        - trajectory exists in raw corpus
        - trajectory is in curation_report.eligible
        - record request matches original trajectory request
        - correction_id was approved as usable by curation

        A trusted dataset promotion operation alone is not enough
        to bypass the raw-evidence provenance boundary.
        """

        raw_trajectories = (
            load_trajectories(
                self.trajectory_path
            )
        )

        raw_by_id = {
            trajectory.trajectory_id:
                trajectory

            for trajectory
            in raw_trajectories
        }

        eligible_by_id = {
            item.trajectory_id:
                item

            for item
            in curation_report.eligible
        }

        # ----------------------------------------------------
        # Trajectory lineage
        # ----------------------------------------------------

        invalid_trajectory_ids: list[
            str
        ] = []

        for record in records:

            if (
                record.trajectory_id
                not in raw_by_id
                or record.trajectory_id
                not in eligible_by_id
            ):

                invalid_trajectory_ids.append(
                    record.trajectory_id
                )

        if invalid_trajectory_ids:

            unique_ids = (
                sorted(
                    set(
                        invalid_trajectory_ids
                    )
                )
            )

            raise ValueError(
                "Dataset contains trajectory_id "
                "not present in "
                "curation_report.eligible: "
                + ", ".join(
                    unique_ids
                )
            )

        # ----------------------------------------------------
        # Request lineage
        # ----------------------------------------------------

        mismatched_requests: list[
            str
        ] = []

        for record in records:

            trajectory = (
                raw_by_id[
                    record.trajectory_id
                ]
            )

            record_request = (
                normalize_request(
                    record.user_request
                )
            )

            trajectory_request = (
                normalize_request(
                    trajectory.user_request
                )
            )

            if (
                record_request
                != trajectory_request
            ):

                mismatched_requests.append(
                    record.record_id
                )

        if mismatched_requests:

            raise ValueError(
                "Dataset user_request does not "
                "match its source trajectory for "
                "record_id: "
                + ", ".join(
                    sorted(
                        mismatched_requests
                    )
                )
            )

        # ----------------------------------------------------
        # Correction lineage
        # ----------------------------------------------------

        invalid_corrections: list[
            str
        ] = []

        for record in records:

            curated = (
                eligible_by_id[
                    record.trajectory_id
                ]
            )

            if (
                record.correction_id
                not in curated
                .usable_correction_ids
            ):

                invalid_corrections.append(
                    record.correction_id
                )

        if invalid_corrections:

            raise ValueError(
                "Dataset contains correction_id "
                "not approved by curation: "
                + ", ".join(
                    sorted(
                        set(
                            invalid_corrections
                        )
                    )
                )
            )

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        *,
        dataset_version: str,
        validation_fraction: float = 0.20,
        split_seed: str = (
            DEFAULT_SPLIT_SEED
        ),
        diversity_policy: (
            DiversityGatePolicy
            | None
        ) = None,
        eval_paths: (
            Iterable[
                Path
            ]
            | None
        ) = None,
    ) -> TrustedTrainingPipelineResult:

        resolved_eval_paths = (
            self._resolve_eval_paths(
                eval_paths
            )
        )

        # ====================================================
        # PHASE 3B
        # ====================================================

        curation_report = (
            curate_corpus(
                trajectory_path=(
                    self.trajectory_path
                ),

                correction_path=(
                    self.correction_path
                ),

                eval_paths=(
                    resolved_eval_paths
                ),
            )
        )

        self._assert_curation_safe(
            curation_report
        )

        # ====================================================
        # PHASE 3C
        # ====================================================

        diversity_report = (
            evaluate_diversity_gate(
                trajectory_path=(
                    self.trajectory_path
                ),

                curation_report=(
                    curation_report
                ),

                policy=(
                    diversity_policy
                ),
            )
        )

        self._assert_diversity_safe(
            diversity_report
        )

        # ====================================================
        # VERIFIED PROMOTED DATASET
        #
        # PreferenceDatasetLoader performs the existing immutable
        # manifest / hash / schema verification.
        # ====================================================

        (
            source_manifest,
            records,
        ) = (
            self.dataset_loader
            .load(
                dataset_version
            )
        )

        # ====================================================
        # PROVENANCE
        # ====================================================

        self._assert_dataset_lineage(
            records=(
                records
            ),

            curation_report=(
                curation_report
            ),
        )

        # ====================================================
        # PHASE 3D
        #
        # Exporter independently performs held-out intersection
        # checks again before writing anything.
        # ====================================================

        split_manifest = (
            self.exporter
            .export(
                version=(
                    dataset_version
                ),

                validation_fraction=(
                    validation_fraction
                ),

                split_seed=(
                    split_seed
                ),

                eval_paths=(
                    resolved_eval_paths
                ),
            )
        )

        return (
            TrustedTrainingPipelineResult(
                source_dataset_version=(
                    source_manifest
                    .version
                ),

                dataset_record_count=(
                    len(
                        records
                    )
                ),

                curated_eligible_count=(
                    curation_report
                    .eligible_trajectory_count
                ),

                curation=(
                    curation_report
                ),

                diversity=(
                    diversity_report
                ),

                split=(
                    split_manifest
                ),
            )
        )