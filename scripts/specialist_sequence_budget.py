from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from config import Settings
from learning.training.specialist_sft import (
    _render_prompt,
    build_runtime_messages,
    load_external_sft_jsonl,
    verify_specialist_sft_corpus,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--external-sft-jsonl",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1536,
    )
    args = parser.parse_args()

    (
        manifest,
        train,
        validation,
        system_prompt,
    ) = verify_specialist_sft_corpus(
        project_root=PROJECT_ROOT,
        corpus_directory=args.corpus,
    )

    settings = Settings()
    profile = settings.require_model_profile(
        manifest.base_model_key
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(
            profile.model_path
            .expanduser()
            .resolve()
        ),
        local_files_only=True,
    )

    rows = []

    for record in [
        *train,
        *validation,
    ]:
        messages = build_runtime_messages(
            system_prompt=system_prompt,
            user_request=record.user_request,
            semantic_context=record.semantic_context,
        )

        prompt = _render_prompt(
            tokenizer,
            messages,
        )

        prompt_tokens = len(
            tokenizer(
                prompt,
                add_special_tokens=True,
            )["input_ids"]
        )

        target_tokens = len(
            tokenizer(
                record.target_response
                + (tokenizer.eos_token or ""),
                add_special_tokens=False,
            )["input_ids"]
        )

        rows.append(
            (
                "capability",
                record.record_id,
                prompt_tokens,
                target_tokens,
            )
        )

    if args.external_sft_jsonl:
        records, _sources = (
            load_external_sft_jsonl(
                args.external_sft_jsonl
            )
        )

        for record in records:
            prompt = _render_prompt(
                tokenizer,
                record["prompt_messages"],
            )

            prompt_tokens = len(
                tokenizer(
                    prompt,
                    add_special_tokens=True,
                )["input_ids"]
            )

            target_tokens = len(
                tokenizer(
                    record["chosen"]
                    + (tokenizer.eos_token or ""),
                    add_special_tokens=False,
                )["input_ids"]
            )

            rows.append(
                (
                    "external",
                    (
                        f"{record['source_path']}:"
                        f"{record['line_number']}"
                    ),
                    prompt_tokens,
                    target_tokens,
                )
            )

    truncated = [
        row
        for row in rows
        if (
            row[2]
            + row[3]
            > args.max_length
        )
    ]

    minimum_prompt_kept = min(
        (
            args.max_length
            - target
            for _, _, _prompt, target
            in rows
        ),
        default=args.max_length,
    )

    worst = sorted(
        rows,
        key=lambda row: (
            row[2]
            + row[3]
        ),
        reverse=True,
    )[:10]

    print(
        "SPECIALIST SEQUENCE BUDGET DIAGNOSTIC"
    )
    print(
        "====================================="
    )
    print(
        "specialist:",
        manifest.specialist,
    )
    print(
        "prompt profile:",
        manifest.worker_prompt_profile,
    )
    print(
        "max length:",
        args.max_length,
    )
    print(
        "records:",
        len(rows),
    )
    print(
        "would truncate:",
        len(truncated),
    )
    print(
        "minimum prompt tokens retained:",
        minimum_prompt_kept,
    )
    print()
    print(
        "worst combined sequences:"
    )

    for (
        kind,
        name,
        prompt_tokens,
        target_tokens,
    ) in worst:
        print(
            kind,
            "prompt=",
            prompt_tokens,
            "target=",
            target_tokens,
            "combined=",
            prompt_tokens + target_tokens,
            "id=",
            name,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
