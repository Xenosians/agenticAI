from __future__ import annotations

from pathlib import (
    Path,
)

import learning

from learning.datasets import (
    records as dataset_records,
    training_export as dataset_training_export,
)


from learning.cli import (
    run_corpus_analysis,
    run_curation,
    run_dataset_promotion,
    run_diversity_gate,
    run_eval_reports,
    run_gateway_eval,
    run_learning_eval,
    run_promotion_gate,
    run_review,
    run_training_preflight,
)

from learning.paths import (
    CORRECTIONS_PATH,
    DATASETS_ROOT,
    EVALUATIONS_ROOT,
    EVALUATION_SUITE_ROOT,
    LEARNING_ROOT,
    PROMOTIONS_ROOT,
    REPOSITORY_ROOT,
    REVIEWS_PATH,
    RUNTIME_LEARNING_ROOT,
    TRAINING_ROOT,
    TRAJECTORIES_PATH,
)


def test_learning_roots_are_derived_from_package_location(
) -> None:

    package_root = (
        Path(
            learning.__file__
        )
        .resolve()
        .parent
    )

    assert (
        LEARNING_ROOT
        == package_root
    )

    assert (
        REPOSITORY_ROOT
        == package_root.parent
    )

    assert (
        RUNTIME_LEARNING_ROOT
        == (
            REPOSITORY_ROOT
            / ".runtime"
            / "learning"
        )
    )


def test_runtime_learning_paths_are_canonical(
) -> None:

    assert (
        TRAJECTORIES_PATH
        == (
            RUNTIME_LEARNING_ROOT
            / "trajectories.jsonl"
        )
    )

    assert (
        CORRECTIONS_PATH
        == (
            RUNTIME_LEARNING_ROOT
            / "corrections.jsonl"
        )
    )

    assert (
        REVIEWS_PATH
        == (
            RUNTIME_LEARNING_ROOT
            / "reviews.jsonl"
        )
    )

    assert (
        DATASETS_ROOT
        == (
            RUNTIME_LEARNING_ROOT
            / "datasets"
        )
    )

    assert (
        EVALUATIONS_ROOT
        == (
            RUNTIME_LEARNING_ROOT
            / "evaluations"
        )
    )

    assert (
        PROMOTIONS_ROOT
        == (
            RUNTIME_LEARNING_ROOT
            / "promotions"
        )
    )

    assert (
        TRAINING_ROOT
        == (
            RUNTIME_LEARNING_ROOT
            / "training"
        )
    )


def test_evaluation_suite_path_matches_reorganized_package(
) -> None:

    assert (
        EVALUATION_SUITE_ROOT
        == (
            LEARNING_ROOT
            / "evaluation"
            / "evals"
        )
    )

    assert (
        EVALUATION_SUITE_ROOT.name
        == "evals"
    )

    assert (
        EVALUATION_SUITE_ROOT.parent.name
        == "evaluation"
    )


def test_phase4d_cli_defaults_use_canonical_paths(
) -> None:

    assert (
        run_corpus_analysis
        .DEFAULT_TRAJECTORIES
        == TRAJECTORIES_PATH
    )

    assert (
        run_corpus_analysis
        .DEFAULT_CORRECTIONS
        == CORRECTIONS_PATH
    )

    assert (
        run_corpus_analysis
        .DEFAULT_EVAL_DIRECTORY
        == EVALUATION_SUITE_ROOT
    )

    assert (
        run_review
        .DEFAULT_TRAJECTORIES
        == TRAJECTORIES_PATH
    )

    assert (
        run_review
        .DEFAULT_CORRECTIONS
        == CORRECTIONS_PATH
    )

    assert (
        run_review
        .DEFAULT_REVIEWS
        == REVIEWS_PATH
    )

    assert (
        run_curation
        .DEFAULT_TRAJECTORIES
        == TRAJECTORIES_PATH
    )

    assert (
        run_curation
        .DEFAULT_CORRECTIONS
        == CORRECTIONS_PATH
    )

    assert (
        run_curation
        .DEFAULT_REVIEWS
        == REVIEWS_PATH
    )

    assert (
        run_curation
        .DEFAULT_EVAL_DIRECTORY
        == EVALUATION_SUITE_ROOT
    )

    assert (
        run_diversity_gate
        .DEFAULT_TRAJECTORIES
        == TRAJECTORIES_PATH
    )

    assert (
        run_diversity_gate
        .DEFAULT_CORRECTIONS
        == CORRECTIONS_PATH
    )

    assert (
        run_diversity_gate
        .DEFAULT_REVIEWS
        == REVIEWS_PATH
    )

    assert (
        run_diversity_gate
        .DEFAULT_EVAL_DIRECTORY
        == EVALUATION_SUITE_ROOT
    )

    assert (
        run_dataset_promotion
        .DEFAULT_TRAJECTORIES
        == TRAJECTORIES_PATH
    )

    assert (
        run_dataset_promotion
        .DEFAULT_CORRECTIONS
        == CORRECTIONS_PATH
    )

    assert (
        run_dataset_promotion
        .DEFAULT_REVIEWS
        == REVIEWS_PATH
    )

    assert (
        run_dataset_promotion
        .DEFAULT_DATASETS
        == DATASETS_ROOT
    )

    assert (
        run_dataset_promotion
        .DEFAULT_EVAL_DIRECTORY
        == EVALUATION_SUITE_ROOT
    )


def test_remaining_cli_defaults_use_canonical_paths(
) -> None:

    assert (
        run_eval_reports
        .DEFAULT_ROOT
        == EVALUATIONS_ROOT
    )

    assert (
        run_gateway_eval
        .DEFAULT_SUITE
        == (
            EVALUATION_SUITE_ROOT
            / "core.v1.jsonl"
        )
    )

    assert (
        run_gateway_eval
        .DEFAULT_REPORT_ROOT
        == EVALUATIONS_ROOT
    )

    assert (
        run_learning_eval
        .BOOTSTRAP_REPOSITORY_ROOT
        == REPOSITORY_ROOT
    )

    assert (
        run_learning_eval
        .DEFAULT_SUITE
        == (
            EVALUATION_SUITE_ROOT
            / "core.v1.jsonl"
        )
    )

    assert (
        run_learning_eval
        .DEFAULT_REPORT_ROOT
        == EVALUATIONS_ROOT
    )

    assert (
        run_promotion_gate
        .DEFAULT_REPORT_ROOT
        == EVALUATIONS_ROOT
    )

    assert (
        run_promotion_gate
        .DEFAULT_PROMOTION_ROOT
        == PROMOTIONS_ROOT
    )

    assert (
        run_training_preflight
        .DEFAULT_OUTPUT_ROOT
        == TRAINING_ROOT
    )


def test_dataset_defaults_use_canonical_evaluation_suite_path(
) -> None:

    assert (
        dataset_records
        .DEFAULT_EVAL_DIRECTORY
        == EVALUATION_SUITE_ROOT
    )

    assert (
        dataset_training_export
        .DEFAULT_EVAL_DIRECTORY
        == EVALUATION_SUITE_ROOT
    )

