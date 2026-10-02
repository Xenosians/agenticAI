from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning.training.specialist_sft import load_training_config


def load_report(path: Path) -> dict:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "specialist-model-evaluation.v1":
        raise SystemExit(f"Invalid specialist evaluation report: {path}")
    return payload


def case_map(report: dict) -> dict[str, bool]:
    result = {}
    for item in report.get("cases", []):
        if not isinstance(item, dict):
            raise SystemExit("Invalid evaluation case.")
        name = item.get("name")
        passed = item.get("passed")
        if not isinstance(name, str) or not isinstance(passed, bool):
            raise SystemExit("Invalid evaluation case result.")
        result[name] = passed
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Explicitly promote an already-gated specialist candidate.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--allow-promotion", action="store_true")
    args = parser.parse_args()

    if not args.allow_promotion:
        raise SystemExit("Refusing to modify production binding without --allow-promotion.")

    config = load_training_config(args.config)
    baseline = load_report(args.baseline_report)
    candidate = load_report(args.candidate_report)
    if baseline.get("agent_name") != config.specialist or candidate.get("agent_name") != config.specialist:
        raise SystemExit("Evaluation report targets the wrong specialist.")
    if candidate.get("model_key") != config.candidate_model_key:
        raise SystemExit("Candidate report model key does not match config.")
    if candidate.get("promotion_gate_passed") is not True:
        raise SystemExit("Candidate protocol gate did not pass.")

    before = case_map(baseline)
    after = case_map(candidate)
    if set(before) != set(after):
        raise SystemExit("Baseline and candidate reports use different cases.")
    regressions = [name for name, passed in before.items() if passed and not after[name]]
    if regressions:
        raise SystemExit(f"Candidate has regressions: {regressions}")
    if float(candidate.get("pass_rate", 0.0)) < float(baseline.get("pass_rate", 0.0)):
        raise SystemExit("Candidate pass rate is below baseline.")

    agent_file = PROJECT_ROOT / config.agent_definition
    text = agent_file.read_text(encoding="utf-8")
    pattern = re.compile(r"(?m)^model:\s*(\S+)\s*$")
    match = pattern.search(text)
    if match is None:
        raise SystemExit(f"No model binding found in {agent_file}.")
    current = match.group(1)
    if current == config.candidate_model_key:
        print("SPECIALIST PROMOTION: already promoted")
        return 0
    if current != config.base_model_key:
        raise SystemExit(
            "Current production model does not match configured baseline: "
            f"{current!r} != {config.base_model_key!r}"
        )

    agent_file.write_text(
        pattern.sub(f"model: {config.candidate_model_key}", text, count=1),
        encoding="utf-8",
    )
    print("SPECIALIST PROMOTION: PASS")
    print("specialist:", config.specialist)
    print("production model:", config.candidate_model_key)
    print("Git is the rollback mechanism; no backup file was created.")
    print("No provider execution occurred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
