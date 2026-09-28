from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = (
    Path(__file__).resolve().parents[3]
)

DEFAULT_ACTIVE_POINTER = (
    REPOSITORY_ROOT
    / ".runtime"
    / "learning"
    / "continual"
    / "active-checkpoint.json"
)

DEFAULT_CHECKPOINT_ROOT = (
    REPOSITORY_ROOT
    / ".runtime"
    / "learning"
    / "continual"
    / "checkpoints"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return value


def _fingerprint_directory(
    directory: Path,
) -> str:
    directory = (
        directory.expanduser().resolve()
    )

    if not directory.is_dir():
        raise ValueError(
            f"Directory does not exist: {directory}"
        )

    digest = hashlib.sha256()
    count = 0

    for path in sorted(
        (
            candidate
            for candidate in directory.rglob("*")
            if candidate.is_file()
        ),
        key=lambda value:
            str(
                value.relative_to(
                    directory
                )
            ),
    ):
        count += 1

        relative = str(
            path.relative_to(
                directory
            )
        )

        file_digest = hashlib.sha256()

        with path.open("rb") as handle:
            while True:
                chunk = handle.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                file_digest.update(
                    chunk
                )

        digest.update(
            relative.encode("utf-8")
        )
        digest.update(
            file_digest.hexdigest().encode("ascii")
        )
        digest.update(
            str(path.stat().st_size).encode("ascii")
        )

    if count == 0:
        raise ValueError(
            f"Directory is empty: {directory}"
        )

    return digest.hexdigest()


def _resolve_checkpoint_id(
    *,
    checkpoint_id_override: str | None,
    active_pointer_path: Path,
) -> tuple[str | None, bool]:
    if checkpoint_id_override is not None:
        normalized = (
            checkpoint_id_override.strip()
        )

        if not normalized:
            raise ValueError(
                "checkpoint_id_override must not be empty."
            )

        return normalized, True

    if not active_pointer_path.is_file():
        return None, False

    pointer = _read_json(
        active_pointer_path
    )

    checkpoint_id = pointer.get(
        "checkpoint_id"
    )

    if (
        not isinstance(checkpoint_id, str)
        or not checkpoint_id.strip()
    ):
        raise ValueError(
            "Active checkpoint pointer is malformed."
        )

    return checkpoint_id.strip(), False


def apply_phase5_adapter_overlay(
    *,
    backend,
    model_key: str,
    base_model_path: Path,
    checkpoint_id_override: str | None = None,
    allow_unpromoted_override: bool = False,
    active_pointer_path: Path = DEFAULT_ACTIVE_POINTER,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
):
    """
    Apply one verified Phase-5 LoRA adapter to an already-loaded backend.

    Production accepts only the promotion-gated active pointer.
    Isolated evaluation may use an explicit unpromoted checkpoint override.
    """
    checkpoint_id, is_override = (
        _resolve_checkpoint_id(
            checkpoint_id_override=(
                checkpoint_id_override
            ),
            active_pointer_path=(
                active_pointer_path
                .expanduser()
                .resolve()
            ),
        )
    )

    if checkpoint_id is None:
        return backend

    if (
        is_override
        and not allow_unpromoted_override
    ):
        raise PermissionError(
            "Unpromoted checkpoint override is allowed only "
            "for explicit isolated evaluation."
        )

    checkpoint_directory = (
        checkpoint_root
        .expanduser()
        .resolve()
        / checkpoint_id
    )

    manifest_path = (
        checkpoint_directory
        / "manifest.json"
    )

    if not manifest_path.is_file():
        raise ValueError(
            "Phase-5 checkpoint manifest does not exist: "
            f"{manifest_path}"
        )

    manifest = _read_json(
        manifest_path
    )

    if manifest.get(
        "target_model_key"
    ) != model_key:
        return backend

    adapter_directory = Path(
        str(
            manifest.get(
                "adapter_directory"
            )
        )
    ).expanduser().resolve()

    expected_adapter_sha = (
        manifest.get(
            "adapter_sha256"
        )
    )

    observed_adapter_sha = (
        _fingerprint_directory(
            adapter_directory
        )
    )

    if observed_adapter_sha != expected_adapter_sha:
        raise ValueError(
            "Phase-5 adapter SHA-256 verification failed."
        )

    observed_base_sha = (
        _fingerprint_directory(
            base_model_path
        )
    )

    if (
        observed_base_sha
        != manifest.get(
            "base_model_sha256"
        )
    ):
        raise ValueError(
            "Phase-5 adapter base-model identity mismatch."
        )

    if not is_override:
        promotion_path = (
            checkpoint_directory
            / "promotion.json"
        )

        if not promotion_path.is_file():
            raise PermissionError(
                "Active Phase-5 checkpoint has no promotion artifact."
            )

        promotion = _read_json(
            promotion_path
        )

        if promotion.get(
            "promotion_eligible"
        ) is not True:
            raise PermissionError(
                "Active Phase-5 checkpoint promotion is not eligible."
            )

    if not hasattr(backend, "model"):
        raise TypeError(
            "Configured backend does not expose a model for PEFT overlay."
        )

    from peft import PeftModel

    backend.model = (
        PeftModel
        .from_pretrained(
            backend.model,
            str(adapter_directory),
            is_trainable=False,
        )
    )

    backend.model.eval()

    print(
        "[MODEL] Applied Phase-5 adapter "
        f"checkpoint='{checkpoint_id}' "
        f"model='{model_key}' "
        f"evaluation_override={is_override}"
    )

    return backend
