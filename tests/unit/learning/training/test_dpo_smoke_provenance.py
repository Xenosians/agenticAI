import json

from pathlib import Path

from learning.cli.run_dpo_qlora_smoke import (
    PROJECT_ROOT,
    _build_model_profile,
    _synthetic_preference_records,
)

from learning.training.dpo_materializer import (
    SpecialistDpoMaterializer,
)


def write_agent(
    tmp_path: Path,
) -> Path:

    path = (
        tmp_path
        / "account-specialist.md"
    )

    path.write_text(
        """---
name: account-specialist
description: Handles account operations.
tools:
  - account_status
  - unlock_user
model: qwen2.5-0.5b-funccall
max_steps: 3
---

You are an account specialist.
""",
        encoding="utf-8",
    )

    return path


def write_model(
    tmp_path: Path,
) -> Path:

    model_path = (
        tmp_path
        / "model"
    )

    model_path.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        model_path
        / "config.json"
    ).write_text(
        json.dumps(
            {
                "model_type":
                    "synthetic-qwen",
            }
        ),
        encoding="utf-8",
    )

    (
        model_path
        / "generation_config.json"
    ).write_text(
        json.dumps(
            {
                "max_new_tokens":
                    256,
            }
        ),
        encoding="utf-8",
    )

    (
        model_path
        / "tokenizer.json"
    ).write_text(
        json.dumps(
            {
                "version":
                    "1.0",
            }
        ),
        encoding="utf-8",
    )

    (
        model_path
        / "tokenizer_config.json"
    ).write_text(
        json.dumps(
            {
                "chat_template":
                    (
                        "{% for message in messages %}"
                        "{{ message['role'] }}:"
                        "{{ message['content'] }}"
                        "{% endfor %}"
                    ),
            }
        ),
        encoding="utf-8",
    )

    (
        model_path
        / "model.safetensors"
    ).write_bytes(
        b"synthetic-smoke-model"
    )

    return model_path


def test_project_root_is_repository_root():

    assert (
        PROJECT_ROOT
        / "learning"
        / "cli"
        / "run_dpo_qlora_smoke.py"
    ).is_file()

    assert (
        PROJECT_ROOT
        / "subagents"
        / "agents"
    ).is_dir()


def test_synthetic_smoke_records_have_complete_provenance_and_materialize(
    tmp_path: Path,
):

    agent_path = (
        write_agent(
            tmp_path
        )
    )

    model_path = (
        write_model(
            tmp_path
        )
    )

    model_profile = (
        _build_model_profile(
            model_path
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
                tmp_path
                / "materialized"
            ),
        )
    )

    (
        training_records,
        validation_records,
    ) = (
        _synthetic_preference_records(
            agent=(
                materializer.agent
            ),

            model_profile=(
                model_profile
            ),
        )
    )

    assert (
        len(
            training_records
        )
        == 2
    )

    assert (
        len(
            validation_records
        )
        == 1
    )

    for record in (
        training_records
        + validation_records
    ):

        assert (
            record.execution_provenance
            is not None
        )

        assert (
            record
            .execution_provenance
            .provenance_complete
            is True
        )

        assert (
            record
            .execution_provenance
            .agent_name
            == "account-specialist"
        )

        assert (
            record
            .execution_provenance
            .model_key
            == "qwen2.5-0.5b-funccall"
        )

    train_result = (
        materializer.build(
            records=(
                training_records
            ),

            source_split_id=(
                "synthetic-smoke-test"
            ),

            source_partition=(
                "train"
            ),

            source_sha256=(
                "synthetic-train"
            ),
        )
    )

    validation_result = (
        materializer.build(
            records=(
                validation_records
            ),

            source_split_id=(
                "synthetic-smoke-test"
            ),

            source_partition=(
                "validation"
            ),

            source_sha256=(
                "synthetic-validation"
            ),
        )
    )

    assert (
        train_result
        .manifest
        .execution_provenance_enforced
        is True
    )

    assert (
        validation_result
        .manifest
        .execution_provenance_enforced
        is True
    )

    assert (
        train_result
        .manifest
        .target_model_artifact_sha256
        == validation_result
        .manifest
        .target_model_artifact_sha256
    )
