from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning.training.specialist_sft import register_candidate_profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Register a merged specialist checkpoint as a logical candidate profile."
    )
    parser.add_argument("--training-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    training_manifest = args.training_manifest
    if training_manifest is None:
        if args.output_root is None:
            raise SystemExit("Pass --training-manifest or --output-root.")
        latest = args.output_root / "latest-training.json"
        if not latest.is_file():
            raise SystemExit(f"Missing latest-training.json: {latest}")
        payload = json.loads(latest.read_text(encoding="utf-8"))
        training_manifest = Path(payload["training_manifest"])

    profile = register_candidate_profile(
        env_path=PROJECT_ROOT / ".env",
        training_manifest_path=training_manifest,
    )
    print("SPECIALIST CANDIDATE REGISTRATION: PASS")
    print("backend:", profile.get("backend"))
    print("model_path:", profile.get("model_path"))
    print("Production specialist binding was NOT modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
