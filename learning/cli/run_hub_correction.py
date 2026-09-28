from __future__ import annotations

import argparse
from pathlib import Path

from learning.evidence.hub_routing import DEFAULT_HUB_ROUTING_LEDGER
from learning.training.hub_preferences import (
    HUB_CORRECTIONS_PATH,
    HubCorrectionRecorder,
    find_hub_routing_attempt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record a human-authored correction for one exact historical "
            "Hub routing attempt. This does not make the correction trainable."
        )
    )

    parser.add_argument("trajectory_id")
    parser.add_argument("attempt_id")
    parser.add_argument(
        "--chosen-file",
        type=Path,
        required=True,
        help="JSON file containing the complete chosen Hub router response.",
    )
    parser.add_argument(
        "--note",
        default=None,
    )
    parser.add_argument(
        "--routing-ledger",
        type=Path,
        default=DEFAULT_HUB_ROUTING_LEDGER,
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        default=HUB_CORRECTIONS_PATH,
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    chosen_path = args.chosen_file.expanduser().resolve()

    if not chosen_path.is_file():
        print(
            "ERROR: chosen file does not exist: "
            f"{chosen_path}"
        )
        return 1

    try:
        attempt = find_hub_routing_attempt(
            trajectory_id=args.trajectory_id,
            attempt_id=args.attempt_id,
            path=args.routing_ledger,
        )

        recorder = HubCorrectionRecorder(
            path=args.corrections,
            enabled=True,
        )

        correction = recorder.record(
            trajectory_id=args.trajectory_id,
            attempt=attempt,
            chosen_response=chosen_path.read_text(
                encoding="utf-8"
            ),
            source="explicit_user",
            note=args.note,
        )

        if correction is None:
            raise RuntimeError(
                "Hub correction recorder unexpectedly disabled."
            )

    except Exception as exc:
        print(
            f"ERROR: {exc}"
        )
        return 1

    print("Hub Correction Recorded")
    print("=======================")
    print(
        f"Correction:   {correction.correction_id}"
    )
    print(
        f"Trajectory:   {correction.trajectory_id}"
    )
    print(
        f"Attempt:      {correction.attempt_id}"
    )
    print(
        f"Rejected SHA: {correction.rejected_response_sha256}"
    )
    print(
        f"Chosen SHA:   {correction.chosen_response_sha256}"
    )
    print("Review:       REQUIRED")
    print("Training:     DISABLED")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
