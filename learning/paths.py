from __future__ import annotations

from pathlib import (
    Path,
)


# ============================================================
# REPOSITORY / PACKAGE ROOTS
# ============================================================


LEARNING_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parent
)


REPOSITORY_ROOT = (
    LEARNING_ROOT
    .parent
)


# ============================================================
# RUNTIME LEARNING STORAGE
# ============================================================


RUNTIME_LEARNING_ROOT = (
    REPOSITORY_ROOT
    / ".runtime"
    / "learning"
)


TRAJECTORIES_PATH = (
    RUNTIME_LEARNING_ROOT
    / "trajectories.jsonl"
)


CORRECTIONS_PATH = (
    RUNTIME_LEARNING_ROOT
    / "corrections.jsonl"
)


REVIEWS_PATH = (
    RUNTIME_LEARNING_ROOT
    / "reviews.jsonl"
)


DATASETS_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "datasets"
)


EVALUATIONS_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "evaluations"
)


PROMOTIONS_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "promotions"
)


TRAINING_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "training"
)


# ============================================================
# IMMUTABLE / HELD-OUT EVALUATION SUITES
# ============================================================


EVALUATION_SUITE_ROOT = (
    LEARNING_ROOT
    / "evaluation"
    / "evals"
)
