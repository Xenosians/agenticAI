from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote a specialist logical model key from passing gate evidence."
    )
    parser.add_argument("--agent-definition", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    agent_path = args.agent_definition
    if not agent_path.is_absolute():
        agent_path = PROJECT_ROOT / agent_path
    agent_path = agent_path.resolve()
    evidence = args.evidence.expanduser().resolve()

    report = json.loads(evidence.read_text(encoding="utf-8"))
    if report.get("promotion_gate_passed") is not True:
        raise SystemExit("Refusing promotion: evaluation report did not pass.")

    text = agent_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SystemExit(f"No YAML frontmatter: {agent_path}")

    frontmatter_end = next(
        (i for i in range(1, len(lines)) if lines[i].strip() == "---"),
        None,
    )
    if frontmatter_end is None:
        raise SystemExit(f"Unterminated frontmatter: {agent_path}")

    model_line = next(
        (i for i in range(1, frontmatter_end) if lines[i].startswith("model:")),
        None,
    )
    if model_line is None:
        raise SystemExit(f"No model field: {agent_path}")

    previous = lines[model_line].split(":", 1)[1].strip()
    if previous == args.model_key:
        print(f"Already promoted: {agent_path.name} -> {args.model_key}")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_dir = PROJECT_ROOT / ".runtime/learning/promotions/backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{agent_path.name}.{stamp}.bak"
    backup.write_text(text, encoding="utf-8")

    lines[model_line] = f"model: {args.model_key}"
    agent_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    record = {
        "schema": "specialist-profile-promotion.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "agent_definition": str(agent_path),
        "previous_model_key": previous,
        "promoted_model_key": args.model_key,
        "evaluation_report": str(evidence),
        "evaluation_report_sha256": _sha256(evidence),
        "model_output_is_authorization": False,
    }
    record_path = PROJECT_ROOT / ".runtime/learning/promotions" / f"{agent_path.stem}-{stamp}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"PROMOTED {agent_path.name}: {previous} -> {args.model_key}")
    print("evidence:", evidence)
    print("record:", record_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
