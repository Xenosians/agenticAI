from __future__ import annotations

import json
from pathlib import Path

from learning.continual.checkpoints import (
    AdapterCheckpointStore,
)
from learning.paths import RUNTIME_LEARNING_ROOT


def _latest_manifest(root: Path):
    if not root.is_dir():
        return None

    candidates = sorted(
        root.glob("*/manifest.json"),
        key=lambda path:
            path.stat().st_mtime,
    )

    if not candidates:
        return None

    path = candidates[-1]

    return (
        path,
        json.loads(
            path.read_text(
                encoding="utf-8"
            )
        ),
    )


def main() -> int:
    materialized = _latest_manifest(
        RUNTIME_LEARNING_ROOT
        / "phase5"
        / "materialized"
    )
    training = _latest_manifest(
        RUNTIME_LEARNING_ROOT
        / "phase5"
        / "training-runs"
    )
    active = AdapterCheckpointStore().active()

    print("Phase-5 Status")
    print("==============")

    if materialized is None:
        print("Materialization: none")
    else:
        _path, value = materialized
        print(
            "Materialization: "
            f"{value.get('materialization_id')} "
            f"ready={value.get('ready_for_training')} "
            f"chapter={value.get('chapter_id')}"
        )

    if training is None:
        print("Training run:    none")
    else:
        _path, value = training
        print(
            "Training run:    "
            f"{value.get('run_id')} "
            f"steps={value.get('optimizer_steps')} "
            "checkpoint="
            f"{value.get('registered_checkpoint_id')}"
        )

    if active is None:
        print("Active adapter:  none")
    else:
        print(
            f"Active adapter:  {active.checkpoint_id}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
