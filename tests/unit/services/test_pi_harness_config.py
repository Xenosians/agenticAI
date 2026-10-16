import json

from services.extensions.harness import (
    _write_pi_model_config,
)


def test_pi_isolated_catalog_registers_selected_model(
    tmp_path,
):
    _write_pi_model_config(
        tmp_path,
        provider="openrouter",
        model_id="openrouter/free",
        max_output_tokens=256,
    )

    path = (
        tmp_path
        / "pi"
        / "models.json"
    )

    assert path.is_file()

    document = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        document["$schema"]
        == "https://pi.dev/schemas/models.schema.json"
    )

    provider = (
        document[
            "providers"
        ][
            "openrouter"
        ]
    )

    assert (
        provider["models"]
        == [
            {
                "id":
                    "openrouter/free",

                "maxTokens":
                    256,
            }
        ]
    )

    # modelOverrides cannot bootstrap an unknown model.
    assert (
        "modelOverrides"
        not in provider
    )

    # Credentials must remain process-only.
    assert (
        "apiKey"
        not in provider
    )

    serialized = (
        path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        "sk-"
        not in serialized
    )


def test_pi_isolated_catalog_supports_nested_model_ids(
    tmp_path,
):
    _write_pi_model_config(
        tmp_path,
        provider="groq",
        model_id="openai/gpt-oss-120b",
        max_output_tokens=768,
    )

    document = json.loads(
        (
            tmp_path
            / "pi"
            / "models.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        document[
            "providers"
        ][
            "groq"
        ][
            "models"
        ][0][
            "id"
        ]
        == "openai/gpt-oss-120b"
    )
