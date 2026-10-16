"""Import an operator-reviewed public JSONL dataset for the existing knowledge tools."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.knowledge.importing import KnowledgeImportError, import_knowledge_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Existing reviewed public JSONL dataset.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for immutable content-addressed snapshots.")
    args = parser.parse_args()
    try:
        report = import_knowledge_snapshot(args.source, args.output_dir).report()
    except (KnowledgeImportError, OSError) as exc:
        # Only KnowledgeImportError has deliberately safe text.
        message = str(exc) if isinstance(exc, KnowledgeImportError) else "The import destination could not be accessed."
        print(json.dumps({"error": "knowledge_import_refused", "message": message}), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
