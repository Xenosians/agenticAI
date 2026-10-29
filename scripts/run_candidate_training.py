"""Run one candidate SFT recipe prepared by the Learning run registry.

Usage: run_candidate_training.py RUN_DIR

Reads RUN_DIR/recipe.json (chosen by the server), trains an isolated adapter
candidate under RUN_DIR/candidate and reports progress to RUN_DIR/status.json.
The candidate is never registered, served or promoted here. Errors are reported
as one known code; details go only to the run's private trainer log.
"""
from __future__ import annotations

import json
import math
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learning.training.recipes import AdaptationRecipe, train_recipe  # noqa: E402


def write_status(run_dir: Path, **status) -> None:
    temporary = run_dir / "status.json.tmp"
    temporary.write_text(json.dumps(status, sort_keys=True))
    os.replace(temporary, run_dir / "status.json")


def error_code(exc: BaseException) -> str:
    text = str(exc)
    if "VRAM" in text or "out of memory" in text.lower():
        return "insufficient_gpu_memory"
    if "requires CUDA" in text:
        return "cuda_unavailable"
    if "loss boundary" in text:
        return "chat_template_unsupported"
    if "sequence budget" in text:
        return "example_too_long"
    if "Dataset" in text or "dataset" in text:
        return "dataset_admission_failed"
    return "training_failed"


def finite(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def main() -> int:
    run_dir = Path(sys.argv[1]).resolve()
    recipe = AdaptationRecipe.model_validate_json((run_dir / "recipe.json").read_text())
    progress = {"step": 0, "max_steps": recipe.max_steps, "train_loss": None, "validation_loss": None}
    write_status(run_dir, state="running", **progress)

    def report(step: int, train_loss=None, validation_loss=None) -> None:
        progress["step"] = min(step, recipe.max_steps)
        if finite(train_loss) is not None:
            progress["train_loss"] = train_loss
        if finite(validation_loss) is not None:
            progress["validation_loss"] = validation_loss
        write_status(run_dir, state="running", **progress)

    try:
        candidate = train_recipe(recipe, allow_training=True, progress=report)
        manifest = json.loads((candidate / "manifest.json").read_text())
        progress["step"] = recipe.max_steps
        progress["validation_loss"] = finite(manifest["validation_metrics"].get("eval_loss"))
        progress["train_loss"] = finite(manifest["train_metrics"].get("train_loss")) or progress["train_loss"]
        write_status(run_dir, state="completed", candidate_sha256=manifest["artifact_sha256"], **progress)
        return 0
    except Exception as exc:  # noqa: BLE001 - every failure is reported as one code
        traceback.print_exc()
        write_status(run_dir, state="failed", error=error_code(exc), **progress)
        return 1


if __name__ == "__main__":
    sys.exit(main())
