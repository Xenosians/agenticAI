import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.training.ppo import SandboxPPOSettings, run_sandbox_sequence_ppo

def main():
    p = argparse.ArgumentParser(description="Simulator-only PPO; independent training episodes, no live tools.")
    p.add_argument("--checkpoint-id", required=True)
    p.add_argument("--cycle-id", required=True)
    p.add_argument("--training-suite", required=True)
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--allow-training", action="store_true")
    p.add_argument("--min-free-vram-gib", type=float, default=12)
    args = p.parse_args()
    if not args.allow_training:
        p.error("PPO requires --allow-training")
    if args.min_free_vram_gib < 2:
        p.error("A positive conservative resource budget is required")
    import torch
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] / 1024**3 < args.min_free_vram_gib:
        raise SystemExit("PPO resource preflight failed; no model loaded.")
    result = run_sandbox_sequence_ppo(source_checkpoint_id=args.checkpoint_id,
        cycle_id=args.cycle_id, suite=args.training_suite,
        settings=SandboxPPOSettings.model_validate_json(args.settings.read_text()))
    print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
