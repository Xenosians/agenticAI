from __future__ import annotations

import argparse
import json
from pathlib import Path

from learning.evidence.hub_routing import (
    DEFAULT_HUB_ROUTING_LEDGER,
    HubRoutingLedgerRecord,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect recent sanitized Hub routing provenance. "
            "This command does not train or promote a model."
        )
    )

    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_HUB_ROUTING_LEDGER,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.limit < 1:
        raise SystemExit("--limit must be positive")

    path = args.path.expanduser().resolve()

    if not path.exists():
        print("No Hub routing provenance ledger exists yet.")
        return 0

    records: list[HubRoutingLedgerRecord] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            records.append(
                HubRoutingLedgerRecord.model_validate(
                    json.loads(line)
                )
            )
        except Exception as exc:
            raise SystemExit(
                f"Invalid Hub provenance at {path}:{line_number}: {exc}"
            ) from exc

    selected = records[-args.limit:]

    if args.json:
        print(
            json.dumps(
                [
                    item.model_dump(
                        mode="json",
                        by_alias=True,
                    )
                    for item in selected
                ],
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0

    print("Hub Routing Provenance")
    print("======================")
    print(f"Ledger:    {path}")
    print(f"Records:   {len(records)}")
    print(f"Showing:   {len(selected)}")
    print()

    for item in selected:
        attempt = item.attempt

        print(
            f"{attempt.attempt_id} "
            f"trajectory={item.trajectory_id} "
            f"mode={attempt.mode} "
            f"validation={attempt.validation_status} "
            f"delegations={attempt.validated_delegation_count} "
            f"model_complete={attempt.model_identity.complete} "
            f"raw_exact={attempt.raw_response_exact}"
        )

        if attempt.validation_error:
            print("  error=" + attempt.validation_error)

    print()
    print("Training:   DISABLED")
    print("Promotion:  DISABLED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
