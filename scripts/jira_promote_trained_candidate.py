from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_REPORT = ROOT / ".runtime/evaluation/jira_specialist_hub-main.json"
CANDIDATE_REPORT = ROOT / ".runtime/evaluation/jira_specialist_jira-func-trained.json"
AGENT_FILE = ROOT / "subagents/agents/jira-specialist.md"


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing evaluation report: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SystemExit(f"Invalid evaluation report: {path}")
    return value


def _case_map(report: dict) -> dict[str, bool]:
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise SystemExit("Evaluation report has no case list.")
    result: dict[str, bool] = {}
    for item in cases:
        if not isinstance(item, dict):
            raise SystemExit("Evaluation report contains an invalid case.")
        name = item.get("name")
        passed = item.get("passed")
        if not isinstance(name, str) or not isinstance(passed, bool):
            raise SystemExit("Evaluation report contains an invalid case result.")
        result[name] = passed
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote the already-gated Jira specialist candidate.")
    parser.add_argument("--allow-promotion", action="store_true")
    args = parser.parse_args()
    if not args.allow_promotion:
        raise SystemExit("Refusing to modify production model binding without --allow-promotion.")

    baseline = _load(BASELINE_REPORT)
    candidate = _load(CANDIDATE_REPORT)

    if candidate.get("model_key") != "jira-func-trained":
        raise SystemExit("Candidate report is not for jira-func-trained.")
    if candidate.get("promotion_gate_passed") is not True:
        raise SystemExit("Candidate protocol gate did not pass.")

    baseline_cases = _case_map(baseline)
    candidate_cases = _case_map(candidate)
    if set(baseline_cases) != set(candidate_cases):
        raise SystemExit("Baseline and candidate reports use different case sets.")

    regressions = [name for name, passed in baseline_cases.items() if passed and not candidate_cases[name]]
    if regressions:
        raise SystemExit(f"Candidate has regressions: {regressions}")

    baseline_rate = float(baseline.get("pass_rate", 0.0))
    candidate_rate = float(candidate.get("pass_rate", 0.0))
    if candidate_rate < baseline_rate:
        raise SystemExit("Candidate pass rate is below baseline.")

    text = AGENT_FILE.read_text()
    if "model: jira-func-trained" in text:
        print("JIRA SPECIALIST PROMOTION: already promoted")
        return
    if text.count("model: hub-main") != 1:
        raise SystemExit("Expected exactly one 'model: hub-main' binding in jira-specialist.md.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = ROOT / ".model_promotion_backup" / stamp / "jira-specialist.md"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(AGENT_FILE, backup)

    AGENT_FILE.write_text(text.replace("model: hub-main", "model: jira-func-trained", 1))
    print("JIRA SPECIALIST PROMOTION: PASS")
    print("production Jira model: jira-func-trained")
    print(f"backup: {backup}")
    print("No Jira provider call occurred.")
    print("No Jira mutation occurred.")


if __name__ == "__main__":
    main()
