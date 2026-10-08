import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.training.dpo_qlora import SpecialistDpoQloraSettings, SpecialistDpoQloraDryRun
from learning.training.preference_training import train_preferences

def main():
    p = argparse.ArgumentParser(description="Train DPO from admitted preference partitions; never auto-promote.")
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--min-free-vram-gib", type=float, default=12)
    p.add_argument("--allow-training", action="store_true")
    args = p.parse_args()
    settings = SpecialistDpoQloraSettings.model_validate_json(args.settings.read_text())
    builder = SpecialistDpoQloraDryRun(train_directory=args.train, validation_directory=args.validation, settings=settings)
    print(train_preferences(builder, max_steps=args.max_steps, allow_training=args.allow_training, min_free_vram_gib=args.min_free_vram_gib))

if __name__ == "__main__":
    main()
