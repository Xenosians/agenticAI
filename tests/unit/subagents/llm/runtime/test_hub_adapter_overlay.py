from pathlib import Path

from subagents.llm.runtime.adapter_overlay import (
    apply_hub_adapter_overlay,
)


class _Backend:
    pass


def test_no_pointer_returns_backend_unchanged(
    tmp_path: Path,
):
    backend = _Backend()

    returned = apply_hub_adapter_overlay(
        backend=backend,
        model_key="hub-main",
        base_model_path=tmp_path,
        active_pointer_path=(
            tmp_path / "missing.json"
        ),
        checkpoint_root=(
            tmp_path / "checkpoints"
        ),
    )

    assert returned is backend


def test_unpromoted_override_requires_explicit_eval_permission(
    tmp_path: Path,
):
    backend = _Backend()

    try:
        apply_hub_adapter_overlay(
            backend=backend,
            model_key="hub-main",
            base_model_path=tmp_path,
            checkpoint_id_override=(
                "checkpoint-test"
            ),
            allow_unpromoted_override=False,
            active_pointer_path=(
                tmp_path / "active.json"
            ),
            checkpoint_root=(
                tmp_path / "checkpoints"
            ),
        )

    except PermissionError as exc:
        assert "isolated evaluation" in str(
            exc
        )

    else:
        raise AssertionError(
            "Unpromoted override was accepted."
        )
