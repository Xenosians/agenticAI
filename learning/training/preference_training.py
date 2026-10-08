"""Bounded real DPO training over the existing reviewed provenance partitions."""
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from learning.training.dpo_qlora import SpecialistDpoQloraDryRun, validate_training_base_model_identity
from learning.continual.storage import fingerprint_directory


def train_preferences(builder: SpecialistDpoQloraDryRun, *, max_steps: int,
                      allow_training: bool = False, min_free_vram_gib: float = 12.0) -> Path:
    if not allow_training:
        raise PermissionError("DPO training requires explicit authorization.")
    if not 1 <= max_steps <= 4000 or min_free_vram_gib < 2:
        raise ValueError("Invalid bounded training/resource budget.")
    if builder.settings.output_root.resolve().is_relative_to(builder.settings.base_model_path.resolve()):
        raise ValueError("DPO candidate output cannot live inside immutable base weights.")
    versions = builder._assert_versions()
    train, validation = builder._load_inputs()
    identity = validate_training_base_model_identity(train=train, validation=validation,
        base_model_path=builder.settings.base_model_path)
    import torch
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] / 1024**3 < min_free_vram_gib:
        raise RuntimeError("DPO resource preflight failed before model construction.")
    run = builder.settings.output_root / datetime.now(timezone.utc).strftime("dpo-%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True, exist_ok=False)
    trainer = None
    try:
        trainer = builder._construct_trainer(train=train, validation=validation)
        trainer.args.max_steps = max_steps
        trainer.args.output_dir = str(run / "trainer")
        training = trainer.train()
        evaluation = trainer.evaluate()
        artifact = run / "adapter"
        trainer.save_model(str(artifact))
        report = {"schema": "reviewed-dpo-training.v1", "promotion_status": "candidate_only",
            "max_steps": max_steps, "base_artifact_sha256": identity.content_sha256,
            "train_sha256": train.manifest.content_sha256,
            "validation_sha256": validation.manifest.content_sha256,
            "versions": versions, "train_metrics": training.metrics,
            "validation_metrics": evaluation, "artifact": str(artifact),
            "artifact_sha256": fingerprint_directory(artifact)}
        (run / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        return run
    finally:
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
