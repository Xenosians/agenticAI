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
    normalize_request,
)

from learning.curation.engine import (
    CurationReport,
    curate_corpus,
)

from learning.datasets.dataset_loader import (
    PreferenceDatasetLoader,
)

from learning.curation.diversity_gate import (
    DiversityGatePolicy,
    DiversityGateReport,
    evaluate_diversity_gate,
)

from learning.datasets.training_export import (
    DEFAULT_EVAL_DIRECTORY,
    DEFAULT_SPLIT_SEED,
    PreferenceTrainingSplitExporter,
    TrainingSplitManifest,
)

from learning.evidence.types import (
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

    quarantined_contamination_count: int

    curation: CurationReport

    # IMPORTANT:
    #
    # This report describes the exact unique trajectory source set
    # represented by the promoted dataset, not the entire curated
    # eligible corpus.
    diversity: DiversityGateReport

    split: TrainingSplitManifest


# ============================================================
# PIPELINE
# ============================================================


class TrustedTrainingPipeline:
    """
    Trusted path from immutable raw learning evidence to an
    exported train/validation split.

    Authority chain:

        raw trajectories / corrections
            ->
        append-only trusted review ledger
            ->
        deterministic curation / quarantine
            ->
        verified promoted dataset
            ->
        provenance validation
            ->
        diversity / balance gate on EXACT dataset source set
            ->
        deterministic training export

    Historical quarantined evidence does not poison future clean
    evidence.

    A large diverse curated corpus also cannot be used to make a
    narrow promoted dataset appear diverse. The diversity gate is
    evaluated against the unique source trajectories represented by
    the requested dataset version.
    """

    def __init__(
        self,
        *,
        trajectory_path: Path,
        correction_path: Path,
        review_path: Path,
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
    # QUARANTINE INTEGRITY
    # ========================================================

    def _assert_quarantine_integrity(
        self,
        report: CurationReport,
    ) -> None:
        """
        Historical contamination is allowed to remain in the raw
        corpus.

        A contaminated trajectory must never simultaneously appear
        in the curated eligible set.
        """

        contaminated_ids = {
            item.trajectory_id

            for item
            in report.contamination_matches
        }

        eligible_ids = {
            item.trajectory_id

            for item
            in report.eligible
        }

        intersection = (
            contaminated_ids
            & eligible_ids
        )

        if not intersection:

            return

        raise ValueError(
            "Curation integrity failure: held-out "
            "contaminated trajectory also appears "
            "in eligible evidence: "
            + ", ".join(
                sorted(
                    intersection
                )
            )
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
            "Diversity/balance gate failed "
            "for the exact promoted dataset source set: "
            f"{failed}"
        )

    # ========================================================
    # CORRECTION TARGET RESOLUTION
    # ========================================================

    def _resolve_correction_step(
        self,
        *,
        trajectory,
        correction,
    ):
        """
        Reconstruct the exact immutable trajectory step targeted by
        a correction.

        This mirrors the ambiguity rules used when preference
        examples are created.

        The training boundary must not trust task_id copied from a
        promoted dataset record without independently resolving the
        source correction against raw evidence.
        """

        if (
            correction.task_id
            is not None
        ):

            for step in (
                trajectory.steps
            ):

                if (
                    step.task_id
                    == correction.task_id
                ):

                    return step

            raise ValueError(
                "Raw correction task_id does not exist "
                "in its source trajectory: "
                f"{correction.correction_id}"
            )

        tool_steps = [
            step

            for step
            in trajectory.steps

            if (
                step.proposed_tool
                is not None
            )
        ]

        if len(
            tool_steps
        ) == 1:

            return (
                tool_steps[
                    0
                ]
            )

        if len(
            trajectory.steps
        ) == 1:

            return (
                trajectory.steps[
                    0
                ]
            )

        if not trajectory.steps:

            return None

        raise ValueError(
            "Raw correction has ambiguous specialist "
            "step lineage: "
            f"{correction.correction_id}"
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
        Every promoted dataset record must independently trace back
        to immutable evidence:

        - an existing raw trajectory
        - a Phase 3B eligible trajectory
        - the same normalized user request
        - an existing raw correction
        - the same trajectory referenced by that correction
        - the exact correction-targeted task
        - the exact specialist task context
        - a correction approved by curation

        The promoted dataset is never trusted as its own provenance
        authority.
        """

        raw_trajectories = (
            load_trajectories(
                self.trajectory_path
            )
        )

        raw_corrections = (
            load_corrections(
                self.correction_path
            )
        )

        raw_by_id = {
            trajectory.trajectory_id:
                trajectory

            for trajectory
            in raw_trajectories
        }

        correction_by_id = {
            correction.correction_id:
                correction

            for correction
            in raw_corrections
        }

        eligible_by_id = {
            item.trajectory_id:
                item

            for item
            in curation_report.eligible
        }

        # ----------------------------------------------------
        # Trajectory existence + curation eligibility
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

            raise ValueError(
                "Dataset contains trajectory_id "
                "not present in "
                "curation_report.eligible: "
                + ", ".join(
                    sorted(
                        set(
                            invalid_trajectory_ids
                        )
                    )
                )
            )

        # ----------------------------------------------------
        # Original user request lineage
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

            if (
                normalize_request(
                    record.user_request
                )
                != normalize_request(
                    trajectory.user_request
                )
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
        # Raw correction existence
        # ----------------------------------------------------

        missing_correction_ids: list[
            str
        ] = []

        for record in records:

            if (
                record.correction_id
                not in correction_by_id
            ):

                missing_correction_ids.append(
                    record.correction_id
                )

        if missing_correction_ids:

            raise ValueError(
                "Dataset references correction_id "
                "missing from raw correction corpus: "
                + ", ".join(
                    sorted(
                        set(
                            missing_correction_ids
                        )
                    )
                )
            )

        # ----------------------------------------------------
        # Correction -> trajectory lineage
        # ----------------------------------------------------

        correction_trajectory_mismatches: list[
            str
        ] = []

        for record in records:

            correction = (
                correction_by_id[
                    record.correction_id
                ]
            )

            if (
                correction.trajectory_id
                != record.trajectory_id
            ):

                correction_trajectory_mismatches.append(
                    record.record_id
                )

        if correction_trajectory_mismatches:

            raise ValueError(
                "Dataset correction trajectory lineage "
                "mismatch for record_id: "
                + ", ".join(
                    sorted(
                        correction_trajectory_mismatches
                    )
                )
            )

        # ----------------------------------------------------
        # Correction-targeted task + prompt context
        # ----------------------------------------------------

        task_mismatches: list[
            str
        ] = []

        context_mismatches: list[
            str
        ] = []

        for record in records:

            trajectory = (
                raw_by_id[
                    record.trajectory_id
                ]
            )

            correction = (
                correction_by_id[
                    record.correction_id
                ]
            )

            source_step = (
                self._resolve_correction_step(
                    trajectory=(
                        trajectory
                    ),

                    correction=(
                        correction
                    ),
                )
            )

            if source_step is None:

                if (
                    record.task_id
                    is not None
                ):

                    task_mismatches.append(
                        record.record_id
                    )

                if (
                    record.task_instructions
                    is not None
                ):

                    context_mismatches.append(
                        record.record_id
                    )

                continue

            if (
                record.task_id
                != source_step.task_id
            ):

                task_mismatches.append(
                    record.record_id
                )

            if (
                record.task_instructions
                != source_step.task_instructions
            ):

                context_mismatches.append(
                    record.record_id
                )

        if task_mismatches:

            raise ValueError(
                "Dataset task_id does not match "
                "the correction-targeted raw trajectory "
                "step for record_id: "
                + ", ".join(
                    sorted(
                        task_mismatches
                    )
                )
            )

        if context_mismatches:

            raise ValueError(
                "Dataset task_instructions does not match "
                "the correction-targeted raw trajectory "
                "step for record_id: "
                + ", ".join(
                    sorted(
                        context_mismatches
                    )
                )
            )

        # ----------------------------------------------------
        # Trusted correction approval lineage
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
    # EXACT DATASET SOURCE CURATION VIEW
    # ========================================================

    def _dataset_source_curation_report(
        self,
        *,
        records: list[
            PreferenceDatasetRecord
        ],
        curation_report: CurationReport,
    ) -> CurationReport:
        """
        Build an internal curation view containing only the unique
        trajectories actually represented by the promoted dataset.

        This prevents a large diverse approved corpus from allowing
        a narrow dataset subset to bypass Phase 3C.

        Example:

            curated eligible:
                20 diverse trajectories

            dataset:
                only 2 trajectories

        The diversity gate must see 2, not 20.

        This helper does not mutate the original curation report.
        """

        dataset_source_ids = {
            record.trajectory_id

            for record
            in records
        }

        if not dataset_source_ids:

            raise ValueError(
                "Promoted dataset contains no "
                "source trajectories."
            )

        eligible_by_id = {
            item.trajectory_id:
                item

            for item
            in curation_report.eligible
        }

        missing = (
            dataset_source_ids
            - set(
                eligible_by_id
            )
        )

        if missing:

            raise ValueError(
                "Dataset source trajectory is not "
                "eligible in the curation report: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        # Preserve original deterministic curation ordering.
        selected = [
            item.model_copy(
                deep=True
            )

            for item
            in curation_report.eligible

            if (
                item.trajectory_id
                in dataset_source_ids
            )
        ]

        if (
            len(
                selected
            )
            != len(
                dataset_source_ids
            )
        ):

            raise ValueError(
                "Dataset source trajectory selection "
                "is internally inconsistent."
            )

        return (
            curation_report.model_copy(
                deep=True,

                update={
                    "eligible":
                        selected,

                    "eligible_trajectory_count":
                        len(
                            selected
                        ),
                },
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
        #
        # Curate the immutable raw corpus first.
        # ====================================================

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

        self._assert_quarantine_integrity(
            curation_report
        )

        # ====================================================
        # VERIFIED PROMOTED DATASET
        #
        # The dataset must exist and pass immutable dataset
        # verification before it can influence Phase 3C.
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
        #
        # Every record must trace back through raw evidence and
        # the trusted curation boundary.
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
        # PHASE 3C
        #
        # CRITICAL:
        #
        # Run diversity against the EXACT unique trajectory
        # source set represented by this dataset version.
        #
        # Do not use all curated eligible trajectories here.
        # ====================================================

        dataset_curation_report = (
            self._dataset_source_curation_report(
                records=(
                    records
                ),

                curation_report=(
                    curation_report
                ),
            )
        )

        diversity_report = (
            evaluate_diversity_gate(
                trajectory_path=(
                    self.trajectory_path
                ),

                curation_report=(
                    dataset_curation_report
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
        # PHASE 3D
        #
        # Export independently checks held-out request
        # intersection again before writing anything.
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
                    source_manifest.version
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

                quarantined_contamination_count=(
                    curation_report
                    .held_out_contamination_count
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