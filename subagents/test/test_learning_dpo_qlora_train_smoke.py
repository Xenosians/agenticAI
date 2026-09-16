from pathlib import (
    Path,
)

import pytest
import torch

from learning.run_dpo_qlora_smoke import (
    DEFAULT_MAX_LENGTH,
    PROJECT_ROOT,
)

from learning.run_dpo_qlora_train_smoke import (
    DEFAULT_TRAINING_OUTPUT_ROOT,
    _hash_trainable_parameters,
    authorize_one_step_training,
    build_parser,
)


def test_training_gate_requires_explicit_authorization(
):

    with pytest.raises(
        PermissionError,
        match=(
            "--allow-training"
        ),
    ):

        authorize_one_step_training(
            allow_training=False,
            max_steps=1,
        )


def test_training_gate_requires_max_steps(
):

    with pytest.raises(
        ValueError,
        match=(
            "exactly one optimizer step"
        ),
    ):

        authorize_one_step_training(
            allow_training=True,
            max_steps=None,
        )


@pytest.mark.parametrize(
    "max_steps",
    [
        0,
        2,
        10,
        -1,
    ],
)
def test_training_gate_refuses_any_step_count_except_one(
    max_steps: int,
):

    with pytest.raises(
        ValueError,
        match=(
            "exactly one optimizer step"
        ),
    ):

        authorize_one_step_training(
            allow_training=True,
            max_steps=(
                max_steps
            ),
        )


def test_training_gate_accepts_exactly_one_step(
):

    authorize_one_step_training(
        allow_training=True,
        max_steps=1,
    )


def test_cli_does_not_authorize_training_by_default(
):

    args = (
        build_parser()
        .parse_args(
            []
        )
    )

    assert (
        args.allow_training
        is False
    )

    assert (
        args.max_steps
        is None
    )


def test_cli_preserves_account_specialist_context_length(
):

    args = (
        build_parser()
        .parse_args(
            []
        )
    )

    assert (
        args.max_length
        == DEFAULT_MAX_LENGTH
    )

    assert (
        args.max_length
        == 1536
    )


def test_training_output_is_isolated_under_runtime(
):

    runtime_root = (
        PROJECT_ROOT
        / ".runtime"
    )

    relative = (
        DEFAULT_TRAINING_OUTPUT_ROOT
        .resolve()
        .relative_to(
            runtime_root.resolve()
        )
    )

    assert (
        relative
        == Path(
            "learning"
        )
        / "training-step-smoke"
    )


def test_trainable_parameter_hash_changes_when_weight_changes(
):

    model = (
        torch.nn.Linear(
            4,
            4,
            bias=False,
        )
    )

    before = (
        _hash_trainable_parameters(
            model
        )
    )

    with torch.no_grad():

        model.weight[
            0,
            0,
        ] += 1.0

    after = (
        _hash_trainable_parameters(
            model
        )
    )

    assert (
        before
        != after
    )


def test_trainable_parameter_hash_refuses_frozen_model(
):

    model = (
        torch.nn.Linear(
            4,
            4,
            bias=False,
        )
    )

    for parameter in model.parameters():

        parameter.requires_grad = (
            False
        )

    with pytest.raises(
        ValueError,
        match=(
            "no trainable parameters"
        ),
    ):

        _hash_trainable_parameters(
            model
        )