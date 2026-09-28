from __future__ import annotations

import argparse
from pathlib import Path

from config import Settings

from learning.training.phase5_materializer import (
    find_latest_ready_hub_plan,
    materialize_phase5_training,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize one exact ready curriculum-membership plan "
            "into immutable Hub SFT/DPO artifacts."
        )
    )

    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--model-key",
        default=None,
    )

    args = parser.parse_args()

    settings = Settings()

    model_key = (
        args.model_key
        or settings.hub_model_key
    )

    profile = settings.require_model_profile(
        model_key
    )

    if profile.model_path is None:
        raise SystemExit(
            "Selected model profile has no local model_path."
        )

    plan_dir = (
        args.plan_dir
        if args.plan_dir is not None
        else find_latest_ready_hub_plan()
    )

    result = materialize_phase5_training(
        plan_directory=plan_dir,
        target_model_key=model_key,
        base_model_path=profile.model_path,
    )

    manifest = result.manifest

    print("Phase-5 Training Materialization")
    print("================================")
    print(f"ID:          {manifest.materialization_id}")
    print(f"Plan:        {manifest.source_plan_id}")
    print(f"Chapter:     {manifest.chapter_id}")
    print(f"Model:       {manifest.target_model_key}")
    print(
        f"SFT:         train={manifest.sft_train_count} "
        f"validation={manifest.sft_validation_count}"
    )
    print(
        f"DPO:         train={manifest.dpo_train_count} "
        f"validation={manifest.dpo_validation_count}"
    )
    print(
        f"Contract OK: {manifest.contract_validated_record_count}"
    )
    print(f"Ready:       {manifest.ready_for_training}")
    print(f"Output:      {result.output_directory}")
    print("Training:    DISABLED")
    print("Promotion:   DISABLED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
