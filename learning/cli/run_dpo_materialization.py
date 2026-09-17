from __future__ import annotations

import argparse
import hashlib
import json

from pathlib import (
    Path,
)

from config import (
    get_settings,
)

from learning.training.dpo_materializer import (
    SpecialistDpoMaterializer,
)

from learning.evidence.types import (
    PreferenceDatasetRecord,
)

from subagents.core.definitions.loader import (
    load_agent_definition,
)


PROJECT_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        2
    ]
)


DEFAULT_AGENT_DIRECTORY = (
    PROJECT_ROOT
    / "subagents"
    / "agents"
)


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / ".runtime"
    / "learning"
    / "dpo"
)


def sha256_file(
    path: Path,
) -> str:

    digest = (
        hashlib.sha256()
    )

    with path.open(
        "rb"
    ) as handle:

        while True:

            chunk = (
                handle.read(
                    1024
                    * 1024
                )
            )

            if not chunk:

                break

            digest.update(
                chunk
            )

    return (
        digest.hexdigest()
    )


def load_records(
    path: Path,
) -> list[
    PreferenceDatasetRecord
]:

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for (
            line_number,
            line,
        ) in enumerate(
            handle,
            start=1,
        ):

            if not line.strip():

                continue

            try:

                payload = (
                    json.loads(
                        line
                    )
                )

                record = (
                    PreferenceDatasetRecord
                    .model_validate(
                        payload
                    )
                )

            except Exception as exc:

                raise ValueError(
                    "Invalid preference record at "
                    f"{path}:{line_number}: {exc}"
                ) from exc

            records.append(
                record
            )

    return records


def build_parser(
) -> argparse.ArgumentParser:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Materialize one provenance-verified "
                "target-specific specialist DPO partition."
            )
        )
    )

    parser.add_argument(
        "--split-id",
        required=True,
    )

    parser.add_argument(
        "--partition",
        choices=[
            "train",
            "validation",
        ],
        required=True,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--agent",
        required=True,
    )

    parser.add_argument(
        "--agent-directory",
        type=Path,
        default=(
            DEFAULT_AGENT_DIRECTORY
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    return parser


def main(
) -> int:

    args = (
        build_parser()
        .parse_args()
    )

    input_path = (
        args.input
        .expanduser()
        .resolve()
    )

    if not input_path.is_file():

        print(
            "ERROR: input partition does not exist: "
            f"{input_path}"
        )

        return 1

    agent_path = (
        args.agent_directory
        .expanduser()
        .resolve()
        / f"{args.agent}.md"
    )

    if not agent_path.is_file():

        print(
            "ERROR: agent definition does not exist: "
            f"{agent_path}"
        )

        return 1

    try:

        records = (
            load_records(
                input_path
            )
        )

        agent = (
            load_agent_definition(
                agent_path
            )
        )

        settings = (
            get_settings()
        )

        model_profile = (
            settings
            .require_model_profile(
                agent.model
            )
        )

        materializer = (
            SpecialistDpoMaterializer(
                agent_definition_path=(
                    agent_path
                ),

                model_profile=(
                    model_profile
                ),

                output_root=(
                    args.output_root
                ),
            )
        )

        result = (
            materializer.build(
                records=(
                    records
                ),

                source_split_id=(
                    args.split_id
                ),

                source_partition=(
                    args.partition
                ),

                source_sha256=(
                    sha256_file(
                        input_path
                    )
                ),
            )
        )

    except Exception as exc:

        print(
            "ERROR: "
            f"{exc}"
        )

        return 1

    if args.json:

        print(
            json.dumps(
                result.model_dump(
                    mode="json",
                    by_alias=True,
                ),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )

    else:

        manifest = (
            result.manifest
        )

        print(
            "Specialist DPO Materialization"
        )

        print(
            "=============================="
        )

        print(
            f"Agent:       "
            f"{manifest.target_agent}"
        )

        print(
            f"Model:       "
            f"{manifest.target_model_key}"
        )

        print(
            f"Split:       "
            f"{manifest.source_split_id}"
        )

        print(
            f"Partition:   "
            f"{manifest.source_partition}"
        )

        print(
            f"Included:    "
            f"{manifest.record_count}"
        )

        print(
            f"Excluded:    "
            f"{manifest.excluded_record_count}"
        )

        print(
            "Provenance:   enforced"
        )

        print(
            f"SHA-256:     "
            f"{manifest.content_sha256}"
        )

        if (
            manifest.exclusion_reason_counts
        ):

            print(
                "Exclusions:"
            )

            for (
                reason,
                count,
            ) in (
                manifest
                .exclusion_reason_counts
                .items()
            ):

                print(
                    f"  {reason}: {count}"
                )

        print(
            "Output:      "
            f"{result.output_directory}"
        )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )