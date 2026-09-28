from __future__ import annotations

import argparse
import json

from pathlib import Path

from config import Settings

from learning.continual.corpus_materializer import (
    list_corpus_snapshots,
)

from learning.training.developer_corpus_bridge import (
    materialize_developer_sft_snapshot,
)

from learning.training.phase5_hybrid_qlora import (
    _fallback_ids,
    _template_ids,
)

from subagents.core.definitions.loader import (
    load_agent_directory,
)


def _developer_agent(
    settings: Settings,
):
    agents = load_agent_directory(
        settings.agents_dir
    )

    for agent in agents:

        if (
            agent.name
            == "developer-specialist"
        ):
            return agent

    raise RuntimeError(
        "developer-specialist definition not found."
    )


def _latest_snapshot(
    source_id: str,
) -> Path:

    candidates = [
        snapshot

        for snapshot
        in list_corpus_snapshots()

        if (
            snapshot.source_id
            == source_id
            and snapshot.target_component
            == "developer-specialist"
            and snapshot.revision
            and snapshot
            .decontamination_policy_sha256
        )
    ]

    if not candidates:
        raise RuntimeError(
            "No pinned/decontaminated developer snapshot "
            f"found for {source_id!r}."
        )

    latest = candidates[-1]

    return Path(
        latest.output_directory
    )



def _build_token_length_resolver(
    *,
    model_path: Path,
    backend: str,
):

    normalized_backend = (
        backend
        .strip()
        .lower()
    )

    if (
        normalized_backend
        == "ministral"
    ):

        from transformers import (
            MistralCommonBackend,
        )

        tokenizer = (
            MistralCommonBackend
            .from_pretrained(
                str(
                    model_path
                )
            )
        )

    else:

        from transformers import (
            AutoTokenizer,
        )

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                str(
                    model_path
                ),
                local_files_only=True,
            )
        )

    def resolve(
        record,
    ) -> tuple[
        int,
        int,
        int,
    ]:

        full_messages = [
            *record.prompt_messages,
            {
                "role":
                    "assistant",

                "content":
                    record.chosen,
            },
        ]

        try:

            prompt_ids = (
                _template_ids(
                    tokenizer,
                    record.prompt_messages,
                    add_generation_prompt=True,
                )
            )

            full_ids = (
                _template_ids(
                    tokenizer,
                    full_messages,
                    add_generation_prompt=False,
                )
            )

        except Exception:

            prompt_ids = (
                _fallback_ids(
                    tokenizer,
                    record.prompt_messages,
                    add_generation_prompt=True,
                )
            )

            full_ids = (
                _fallback_ids(
                    tokenizer,
                    full_messages,
                    add_generation_prompt=False,
                )
            )

        prompt_ids = (
            prompt_ids.flatten()
        )

        full_ids = (
            full_ids.flatten()
        )

        common = 0

        limit = min(
            int(
                prompt_ids.numel()
            ),
            int(
                full_ids.numel()
            ),
        )

        while (
            common < limit
            and int(
                prompt_ids[
                    common
                ]
            )
            == int(
                full_ids[
                    common
                ]
            )
        ):

            common += 1

        return (
            int(
                prompt_ids.numel()
            ),
            int(
                full_ids.numel()
            ),
            common,
        )

    return resolve


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Bridge one governed developer corpus snapshot into "
            "Phase-5-compatible SFT material. No optimizer runs."
        )
    )

    parser.add_argument(
        "--source-id",
        default="swe-rebench-v2",
    )

    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max-sequence-tokens",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    args = parser.parse_args()

    settings = Settings()

    developer = _developer_agent(
        settings
    )

    profile = (
        settings
        .require_model_profile(
            developer.model
        )
    )

    if profile.model_path is None:
        raise RuntimeError(
            "Developer model profile has no local model_path."
        )

    snapshot_directory = (
        args.snapshot_dir
        if args.snapshot_dir
        is not None
        else _latest_snapshot(
            args.source_id
        )
    )

    target_contract = (
        Path(
            settings.agents_dir
        )
        / "developer-specialist.md"
    )

    token_length_resolver = (
        _build_token_length_resolver(
            model_path=(
                Path(
                    profile.model_path
                )
            ),
            backend=(
                profile.backend
            ),
        )
    )

    result = (
        materialize_developer_sft_snapshot(
            snapshot_directory=(
                snapshot_directory
            ),

            target_model_key=(
                developer.model
            ),

            base_model_path=(
                Path(
                    profile.model_path
                )
            ),

            target_contract_path=(
                target_contract
            ),

            token_length_resolver=(
                token_length_resolver
            ),

            max_sequence_tokens=(
                args.max_sequence_tokens
            ),

            max_records=(
                args.max_records
            ),
        )
    )

    if args.json:

        print(
            json.dumps(
                result.model_dump(
                    mode="json"
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )

        return 0

    manifest = result.manifest

    print(
        "Developer Corpus SFT Bridge"
    )

    print(
        "==========================="
    )

    print(
        "materialization="
        + manifest.materialization_id
    )

    print(
        "target="
        + manifest.target_component
    )

    print(
        "model="
        + manifest.target_model_key
    )

    print(
        "evaluation="
        + manifest.evaluation_contract
    )

    print(
        "sft_train="
        + str(
            manifest.sft_train_count
        )
    )

    print(
        "sft_validation="
        + str(
            manifest.sft_validation_count
        )
    )

    print(
        "ready_for_training="
        + str(
            manifest.ready_for_training
        )
    )

    print(
        "training_authorized="
        + str(
            manifest.training_authorized
        )
    )

    print(
        "promotion_authorized="
        + str(
            manifest.promotion_authorized
        )
    )

    print(
        "output="
        + result.output_directory
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
