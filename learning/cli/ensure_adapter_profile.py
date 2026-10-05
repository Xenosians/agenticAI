from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _load_profiles(env_path: Path):
    lines = env_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("MODEL_PROFILES="):
            raw = line.split("=", 1)[1].strip()
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
                raw = raw[1:-1]
            return lines, index, json.loads(raw)
    raise ValueError("MODEL_PROFILES is missing from .env")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--base-key", required=True)
    parser.add_argument("--candidate-key", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    args = parser.parse_args()

    env_path = args.env.expanduser().resolve()
    adapter = args.adapter.expanduser().resolve()

    if not adapter.is_dir():
        raise SystemExit(f"Adapter directory missing: {adapter}")
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"adapter_config.json missing: {adapter}")
    if not any((adapter / name).is_file() for name in (
        "adapter_model.safetensors", "adapter_model.bin"
    )):
        raise SystemExit(f"Adapter weights missing: {adapter}")

    lines, index, profiles = _load_profiles(env_path)
    current = profiles.get(args.candidate_key)
    if isinstance(current, dict):
        value = current.get("adapter_path")
        if isinstance(value, str) and Path(value).expanduser().resolve() == adapter:
            print(f"Profile already present: {args.candidate_key}")
            return 0

    base = profiles.get(args.base_key)
    if not isinstance(base, dict):
        raise SystemExit(f"Base profile missing: {args.base_key}")

    candidate = dict(base)
    candidate["adapter_path"] = str(adapter)
    candidate["enabled"] = True
    candidate["worker_prompt_profile"] = "compact"
    profiles[args.candidate_key] = candidate

    lines[index] = "MODEL_PROFILES='" + json.dumps(
        profiles,
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "'"

    temporary = env_path.with_name(".env.ensure-adapter-profile.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, env_path)

    print("REGISTERED:", args.candidate_key)
    print("model_path:", candidate.get("model_path"))
    print("adapter_path:", candidate.get("adapter_path"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
