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


def active_hub_overlay_target_model_key(
    *,
    active_pointer_path: Path = DEFAULT_ACTIVE_POINTER,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> str | None:
    """
    Return the logical model key targeted by the currently active Hub
    continual-learning adapter, if one is configured and well-formed.

    ModelManager uses this only as a conservative serving optimization guard;
    promotion/authorization semantics remain owned by apply_hub_adapter_overlay.
    """
    pointer_path = (
        active_pointer_path
        .expanduser()
        .resolve()
    )

    if not pointer_path.is_file():
        return None

    pointer = _read_json(
        pointer_path
    )

    checkpoint_id = pointer.get(
        "checkpoint_id"
    )

    if (
        not isinstance(
            checkpoint_id,
            str,
        )
        or not checkpoint_id.strip()
    ):
        return None

    manifest_path = (
        checkpoint_root
        .expanduser()
        .resolve()
        / checkpoint_id.strip()
        / "manifest.json"
    )

    if not manifest_path.is_file():
        return None

    manifest = _read_json(
        manifest_path
    )

    target = manifest.get(
        "target_model_key"
    )

    if (
        isinstance(
            target,
            str,
        )
        and target.strip()
    ):
        return target.strip()

    return None


def apply_hub_adapter_overlay(
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
    Apply one verified Hub LoRA adapter to an already-loaded backend.

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
            "Hub adapter checkpoint manifest does not exist: "
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
            "Hub adapter SHA-256 verification failed."
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
            "Hub adapter base-model identity mismatch."
        )

    if not is_override:
        promotion_path = (
            checkpoint_directory
            / "promotion.json"
        )

        if not promotion_path.is_file():
            raise PermissionError(
                "Active Hub adapter checkpoint has no promotion artifact."
            )

        promotion = _read_json(
            promotion_path
        )

        if promotion.get(
            "promotion_eligible"
        ) is not True:
            raise PermissionError(
                "Active Hub adapter checkpoint promotion is not eligible."
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

    setattr(
        backend,
        "_hub_overlay_applied",
        True,
    )

    print(
        "[MODEL] Applied Hub adapter "
        f"checkpoint='{checkpoint_id}' "
        f"model='{model_key}' "
        f"evaluation_override={is_override}"
    )

    return backend


def apply_profile_adapter_overlay(
    *,
    backend,
    model_key: str,
    adapter_path: Path | None,
):
    """
    Apply a trusted model-profile PEFT adapter.

    Unlike the Hub continual-learning overlay above, this path is
    deployment configuration for a specific logical model profile.

    Example:

        developer-func-trained
            model_path   = Ministral base
            adapter_path = Developer LoRA

    Promotion policy still lives outside this function.
    """

    # Profile-switchable backends own adapter lifecycle so a logical
    # Hub/Developer switch can reuse one already-loaded physical base.
    activate_profile_adapter = getattr(
        backend,
        "activate_profile_adapter",
        None,
    )

    if callable(
        activate_profile_adapter
    ):
        activate_profile_adapter(
            model_key,
            adapter_path,
        )

        return backend

    if adapter_path is None:
        return backend

    adapter_directory = (
        adapter_path
        .expanduser()
        .resolve()
    )

    if not adapter_directory.is_dir():
        raise ValueError(
            "Configured model adapter directory does not exist: "
            f"{adapter_directory}"
        )

    required_files = [
        adapter_directory
        / "adapter_config.json",

        adapter_directory
        / "adapter_model.safetensors",
    ]

    missing = [
        path.name
        for path in required_files
        if not path.is_file()
    ]

    if missing:
        raise ValueError(
            "Configured model adapter is incomplete: "
            + ", ".join(missing)
        )

    if not hasattr(
        backend,
        "model",
    ):
        raise TypeError(
            "Configured backend does not expose a model "
            "for PEFT adapter overlay."
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

    adapter_sha256 = (
        _fingerprint_directory(
            adapter_directory
        )
    )

    print(
        "[MODEL] Applied profile adapter "
        f"model='{model_key}' "
        f"path='{adapter_directory}' "
        f"sha256='{adapter_sha256[:16]}...'"
    )

    return backend
