from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_report(path: Path) -> dict:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "specialist-model-evaluation.v1":
        raise ValueError(f"Invalid specialist evaluation report: {path}")
    return value


def case_map(report: dict) -> dict[str, dict]:
    return {
        item["name"]: item
        for item in report.get("cases", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare specialist baseline and candidate reports.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_report(args.baseline)
    candidate = load_report(args.candidate)
    if baseline.get("agent_name") != candidate.get("agent_name"):
        raise SystemExit("Reports target different specialists.")
    baseline_cases = case_map(baseline)
    candidate_cases = case_map(candidate)
    if set(baseline_cases) != set(candidate_cases):
        raise SystemExit("Reports contain different evaluation case sets.")

    regressions, improvements = [], []
    for name in sorted(baseline_cases):
        before = bool(baseline_cases[name].get("passed"))
        after = bool(candidate_cases[name].get("passed"))
        if before and not after:
            regressions.append(name)
        if not before and after:
            improvements.append(name)

    baseline_rate = float(baseline.get("pass_rate", 0.0))
    candidate_rate = float(candidate.get("pass_rate", 0.0))
    eligible = (
        candidate.get("promotion_gate_passed") is True
        and not regressions
        and candidate_rate >= baseline_rate
    )

    print("SPECIALIST MODEL COMPARISON")
    print("===========================")
    print("agent:", baseline.get("agent_name"))
    print("baseline:", baseline.get("model_key"))
    print("candidate:", candidate.get("model_key"))
    print("baseline pass rate:", f"{baseline_rate:.3f}")
    print("candidate pass rate:", f"{candidate_rate:.3f}")
    print("delta:", f"{candidate_rate - baseline_rate:+.3f}")
    print("regressions:", regressions)
    print("improvements:", improvements)
    print("promotion eligible:", eligible)
    print("No production binding was modified.")
    return 0 if eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
