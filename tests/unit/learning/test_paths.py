from __future__ import annotations

from pathlib import (
    Path,
)

import learning

from learning.cli import (
    run_corpus_analysis,
    run_curation,
    run_dataset_promotion,
    run_diversity_gate,
    run_review,
)

from learning.paths import (
    CORRECTIONS_PATH,
    DATASETS_ROOT,
    EVALUATION_SUITE_ROOT,
    LEARNING_ROOT,
    REPOSITORY_ROOT,
    REVIEWS_PATH,
    RUNTIME_LEARNING_ROOT,
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
