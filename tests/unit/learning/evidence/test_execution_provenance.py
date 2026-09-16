import json

from pathlib import (
    Path,
)

import pytest

from config import (
    ModelProfileSettings,
)

from learning.evidence.execution_provenance import (
    PROVENANCE_SCHEMA,
    build_specialist_execution_provenance,
    clear_runtime_model_fingerprint_cache,
    fingerprint_runtime_model_artifact,
)

from subagents.core.tooling.capabilities import (
    build_agent_capability_catalog,
)

from subagents.core.definitions.types import (
    AgentDefinition,
)

from subagents.core.tooling.prompt import (
    build_worker_system_prompt,
)


# ============================================================
# TEST MODEL FIXTURE
# ============================================================


def _write_model(
    root: Path,
    *,
    weight_payload: bytes = (
        b"fake-weight-v1"
    ),
    chat_template: str = (
        "{% for message in messages %}"
        "{{ message['content'] }}"
        "{% endfor %}"
    ),
) -> Path:

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        root
        / "config.json"
    ).write_text(
        json.dumps(
            {
                "model_type":
                    "qwen2",

                "architectures":
                    [
                        "Qwen2ForCausalLM",
                    ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    (
        root
        / "generation_config.json"
    ).write_text(
        json.dumps(
            {
                "do_sample":
                    False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    (
        root
        / "tokenizer_config.json"
    ).write_text(
        json.dumps(
            {
                "eos_token":
                    "<|im_end|>",

                "pad_token":
                    "<|PAD_TOKEN|>",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    (
        root
        / "tokenizer.json"
    ).write_text(
        json.dumps(
            {
                "version":
                    "1.0",

                "model": {
                    "type":
                        "BPE",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    (
        root
        / "chat_template.jinja"
    ).write_text(
        chat_template,
        encoding="utf-8",
    )

    (
        root
        / "model.safetensors"
    ).write_bytes(
        weight_payload
    )

    return (
        root
    )


# ============================================================
# AGENT / PROFILE FIXTURES
# ============================================================


def _agent(
) -> AgentDefinition:

    return (
        AgentDefinition(
            name=(
                "account-specialist"
            ),

            description=(
                "Handles account status requests."
            ),

            model=(
                "qwen2.5-0.5b-funccall"
            ),

            tools=[
                "account_status",
            ],

            max_steps=3,

            system_prompt=(
                "You are an account specialist."
            ),
        )
    )


def _profile(
    model_path: Path,
    *,
    quantization: str = "auto",
) -> ModelProfileSettings:

    return (
        ModelProfileSettings(
            backend=(
                "qwen-funccall"
            ),

            model_path=(
                model_path
            ),

            quantization=(
                quantization
            ),

            compute_dtype=(
                "bfloat16"
            ),

            device_map=(
                "auto"
            ),

            bnb_4bit_quant_type=(
                "nf4"
            ),

            bnb_4bit_use_double_quant=True,
        )
    )


def _messages(
    agent: AgentDefinition,
    task_context: str,
) -> list[
    dict[
        str,
        str,
    ]
]:

    return [
        {
            "role":
                "system",

            "content":
                build_worker_system_prompt(
                    agent
                ),
        },

        {
            "role":
                "user",

            "content":
                "Is jdoe locked?",
        },

        {
            "role":
                "user",

            "content":
                (
                    "Additional task context "
                    "from the routing stage:\n"
                    f"{task_context}"
                ),
        },
    ]


# ============================================================
# MODEL FINGERPRINT TESTS
# ============================================================


def test_runtime_model_fingerprint_is_deterministic(
    tmp_path: Path,
):

    clear_runtime_model_fingerprint_cache()

    model_path = (
        _write_model(
            tmp_path
            / "model"
        )
    )

    first = (
        fingerprint_runtime_model_artifact(
            model_path
        )
    )

    second = (
        fingerprint_runtime_model_artifact(
            model_path
        )
    )

    assert (
        first.content_sha256
        == second.content_sha256
    )

    assert (
        first.weights_sha256
        == second.weights_sha256
    )

    assert (
        first.tokenizer_sha256
        == second.tokenizer_sha256
    )

    assert (
        first.chat_template_sha256
        is not None
    )


def test_runtime_model_fingerprint_changes_when_weights_change(
    tmp_path: Path,
):

    clear_runtime_model_fingerprint_cache()

    model_path = (
        _write_model(
            tmp_path
            / "model",

            weight_payload=(
                b"fake-weight-v1"
            ),
        )
    )

    before = (
        fingerprint_runtime_model_artifact(
            model_path
        )
    )

    (
        model_path
        / "model.safetensors"
    ).write_bytes(
        b"fake-weight-v2"
    )

    # Explicit fresh-disk inspection.
    #
    # Cached identity deliberately represents the process-local
    # deployment snapshot.
    after = (
        fingerprint_runtime_model_artifact(
            model_path,
            use_cache=False,
        )
    )

    assert (
        before.weights_sha256
        != after.weights_sha256
    )

    assert (
        before.content_sha256
        != after.content_sha256
    )


def test_runtime_model_fingerprint_changes_when_chat_template_changes(
    tmp_path: Path,
):

    clear_runtime_model_fingerprint_cache()

    model_path = (
        _write_model(
            tmp_path
            / "model"
        )
    )

    before = (
        fingerprint_runtime_model_artifact(
            model_path
        )
    )

    (
        model_path
        / "chat_template.jinja"
    ).write_text(
        "different-template",
        encoding="utf-8",
    )

    # Explicit fresh-disk inspection.
    after = (
        fingerprint_runtime_model_artifact(
            model_path,
            use_cache=False,
        )
    )

    assert (
        before.chat_template_sha256
        != after.chat_template_sha256
    )

    assert (
        before.tokenizer_sha256
        != after.tokenizer_sha256
    )

    assert (
        before.content_sha256
        != after.content_sha256
    )


def test_runtime_model_cache_represents_process_snapshot(
    tmp_path: Path,
):

    clear_runtime_model_fingerprint_cache()

    model_path = (
        _write_model(
            tmp_path
            / "model",

            weight_payload=(
                b"fake-weight-v1"
            ),
        )
    )

    first = (
        fingerprint_runtime_model_artifact(
            model_path
        )
    )

    (
        model_path
        / "model.safetensors"
    ).write_bytes(
        b"fake-weight-v2"
    )

    # Cached call keeps the process deployment snapshot.
    cached = (
        fingerprint_runtime_model_artifact(
            model_path
        )
    )

    # Explicit uncached call examines current disk bytes.
    fresh = (
        fingerprint_runtime_model_artifact(
            model_path,
            use_cache=False,
        )
    )

    assert (
        cached.weights_sha256
        == first.weights_sha256
    )

    assert (
        cached.content_sha256
        == first.content_sha256
    )

    assert (
        fresh.weights_sha256
        != first.weights_sha256
    )

    assert (
        fresh.content_sha256
        != first.content_sha256
    )


def test_targeted_cache_invalidation_rebuilds_snapshot(
    tmp_path: Path,
):

    clear_runtime_model_fingerprint_cache()

    model_path = (
        _write_model(
            tmp_path
            / "model",

            weight_payload=(
                b"fake-weight-v1"
            ),
        )
    )

    before = (
        fingerprint_runtime_model_artifact(
            model_path
        )
    )

    (
        model_path
        / "model.safetensors"
    ).write_bytes(
        b"fake-weight-v2"
    )

    clear_runtime_model_fingerprint_cache(
        model_path
    )

    after = (
        fingerprint_runtime_model_artifact(
            model_path
        )
    )

    assert (
        before.weights_sha256
        != after.weights_sha256
    )

    assert (
        before.content_sha256
        != after.content_sha256
    )


def test_runtime_model_fingerprint_requires_weights(
    tmp_path: Path,
):

    clear_runtime_model_fingerprint_cache()

    model_path = (
        tmp_path
        / "model"
    )

    model_path.mkdir()

    (
        model_path
        / "config.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    (
        model_path
        / "tokenizer.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "weight files"
        ),
    ):

        fingerprint_runtime_model_artifact(
            model_path
        )


# ============================================================
# EXECUTION PROVENANCE TESTS
# ============================================================


def test_execution_provenance_is_deterministic(
    tmp_path: Path,
):

    clear_runtime_model_fingerprint_cache()

    model_path = (
        _write_model(
            tmp_path
            / "model"
        )
    )

    agent = (
        _agent()
    )

    profile = (
        _profile(
            model_path
        )
    )

    capability_catalog = (
        build_agent_capability_catalog(
            agent,
            include_arguments=True,
        )
    )

    task_context = (
        "Check the account state for jdoe."
    )

    messages = (
        _messages(
            agent,
            task_context,
        )
    )

    first = (
        build_specialist_execution_provenance(
            agent=(
                agent
            ),

            model_profile=(
                profile
            ),

            capability_catalog=(
                capability_catalog
            ),

            messages=(
                messages
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                task_context
            ),

            max_new_tokens=256,
        )
    )

    second = (
        build_specialist_execution_provenance(
            agent=(
                agent
            ),

            model_profile=(
                profile
            ),

            capability_catalog=(
                capability_catalog
            ),

            messages=(
                messages
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                task_context
            ),

            max_new_tokens=256,
        )
    )

    assert (
        first
        == second
    )

    assert (
        first.schema_name
        == PROVENANCE_SCHEMA
    )

    assert (
        first.provenance_complete
        is True
    )

    assert (
        first.agent_name
        == "account-specialist"
    )

    assert (
        first.model_key
        == "qwen2.5-0.5b-funccall"
    )

    assert (
        first.backend
        == "qwen-funccall"
    )

    assert (
        first.max_new_tokens
        == 256
    )


def test_execution_provenance_changes_with_task_context(
    tmp_path: Path,
):

    clear_runtime_model_fingerprint_cache()

    model_path = (
        _write_model(
            tmp_path
            / "model"
        )
    )

    agent = (
        _agent()
    )

    profile = (
        _profile(
            model_path
        )
    )

    capabilities = (
        build_agent_capability_catalog(
            agent,
            include_arguments=True,
        )
    )

    first_context = (
        "Check the account state for jdoe."
    )

    second_context = (
        "Inspect jdoe account state."
    )

    first = (
        build_specialist_execution_provenance(
            agent=(
                agent
            ),

            model_profile=(
                profile
            ),

            capability_catalog=(
                capabilities
            ),

            messages=(
                _messages(
                    agent,
                    first_context,
                )
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                first_context
            ),

            max_new_tokens=256,
        )
    )

    second = (
        build_specialist_execution_provenance(
            agent=(
                agent
            ),

            model_profile=(
                profile
            ),

            capability_catalog=(
                capabilities
            ),

            messages=(
                _messages(
                    agent,
                    second_context,
                )
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                second_context
            ),

            max_new_tokens=256,
        )
    )

    assert (
        first.task_instructions_sha256
        != second.task_instructions_sha256
    )

    assert (
        first.messages_sha256
        != second.messages_sha256
    )

    assert (
        first.system_prompt_sha256
        == second.system_prompt_sha256
    )


def test_execution_provenance_changes_with_model_profile(
    tmp_path: Path,
):

    clear_runtime_model_fingerprint_cache()

    model_path = (
        _write_model(
            tmp_path
            / "model"
        )
    )

    agent = (
        _agent()
    )

    capabilities = (
        build_agent_capability_catalog(
            agent,
            include_arguments=True,
        )
    )

    task_context = (
        "Check the account state for jdoe."
    )

    messages = (
        _messages(
            agent,
            task_context,
        )
    )

    first = (
        build_specialist_execution_provenance(
            agent=(
                agent
            ),

            model_profile=(
                _profile(
                    model_path,
                    quantization="auto",
                )
            ),

            capability_catalog=(
                capabilities
            ),

            messages=(
                messages
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                task_context
            ),

            max_new_tokens=256,
        )
    )

    second = (
        build_specialist_execution_provenance(
            agent=(
                agent
            ),

            model_profile=(
                _profile(
                    model_path,
                    quantization="bnb4",
                )
            ),

            capability_catalog=(
                capabilities
            ),

            messages=(
                messages
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                task_context
            ),

            max_new_tokens=256,
        )
    )

    assert (
        first.model_artifact_sha256
        == second.model_artifact_sha256
    )

    assert (
        first.model_weights_sha256
        == second.model_weights_sha256
    )

    assert (
        first.model_profile_sha256
        != second.model_profile_sha256
    )


def test_execution_provenance_changes_with_capability_catalog(
    tmp_path: Path,
):

    clear_runtime_model_fingerprint_cache()

    model_path = (
        _write_model(
            tmp_path
            / "model"
        )
    )

    agent = (
        _agent()
    )

    profile = (
        _profile(
            model_path
        )
    )

    task_context = (
        "Check the account state for jdoe."
    )

    messages = (
        _messages(
            agent,
            task_context,
        )
    )

    first_catalog = [
        {
            "name":
                "account_status",

            "description":
                "Status lookup.",

            "argument_schema": {
                "user_id": {
                    "type":
                        "str",
                }
            },
        }
    ]

    second_catalog = [
        {
            "name":
                "account_status",

            "description":
                "Different status metadata.",

            "argument_schema": {
                "user_id": {
                    "type":
                        "str",
                }
            },
        }
    ]

    first = (
        build_specialist_execution_provenance(
            agent=(
                agent
            ),

            model_profile=(
                profile
            ),

            capability_catalog=(
                first_catalog
            ),

            messages=(
                messages
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                task_context
            ),

            max_new_tokens=256,
        )
    )

    second = (
        build_specialist_execution_provenance(
            agent=(
                agent
            ),

            model_profile=(
                profile
            ),

            capability_catalog=(
                second_catalog
            ),

            messages=(
                messages
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                task_context
            ),

            max_new_tokens=256,
        )
    )

    assert (
        first.capability_catalog_sha256
        != second.capability_catalog_sha256
    )

    # The supplied model messages did not change.
    assert (
        first.messages_sha256
        == second.messages_sha256
    )


def test_execution_provenance_changes_with_system_prompt(
    tmp_path: Path,
):

    clear_runtime_model_fingerprint_cache()

    model_path = (
        _write_model(
            tmp_path
            / "model"
        )
    )

    profile = (
        _profile(
            model_path
        )
    )

    first_agent = (
        _agent()
    )

    second_agent = (
        _agent()
    )

    second_agent.system_prompt = (
        "You are a different account specialist."
    )

    first_capabilities = (
        build_agent_capability_catalog(
            first_agent,
            include_arguments=True,
        )
    )

    second_capabilities = (
        build_agent_capability_catalog(
            second_agent,
            include_arguments=True,
        )
    )

    task_context = (
        "Check the account state for jdoe."
    )

    first = (
        build_specialist_execution_provenance(
            agent=(
                first_agent
            ),

            model_profile=(
                profile
            ),

            capability_catalog=(
                first_capabilities
            ),

            messages=(
                _messages(
                    first_agent,
                    task_context,
                )
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                task_context
            ),

            max_new_tokens=256,
        )
    )

    second = (
        build_specialist_execution_provenance(
            agent=(
                second_agent
            ),

            model_profile=(
                profile
            ),

            capability_catalog=(
                second_capabilities
            ),

            messages=(
                _messages(
                    second_agent,
                    task_context,
                )
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                task_context
            ),

            max_new_tokens=256,
        )
    )

    assert (
        first.agent_definition_sha256
        != second.agent_definition_sha256
    )

    assert (
        first.system_prompt_sha256
        != second.system_prompt_sha256
    )

    assert (
        first.messages_sha256
        != second.messages_sha256
    )


def test_execution_provenance_normalizes_task_context(
    tmp_path: Path,
):

    clear_runtime_model_fingerprint_cache()

    model_path = (
        _write_model(
            tmp_path
            / "model"
        )
    )

    agent = (
        _agent()
    )

    profile = (
        _profile(
            model_path
        )
    )

    capabilities = (
        build_agent_capability_catalog(
            agent,
            include_arguments=True,
        )
    )

    normalized_context = (
        "Check the account state for jdoe."
    )

    provenance = (
        build_specialist_execution_provenance(
            agent=(
                agent
            ),

            model_profile=(
                profile
            ),

            capability_catalog=(
                capabilities
            ),

            messages=(
                _messages(
                    agent,
                    normalized_context,
                )
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                "   Check the account state for jdoe.   "
            ),

            max_new_tokens=256,
        )
    )

    comparison = (
        build_specialist_execution_provenance(
            agent=(
                agent
            ),

            model_profile=(
                profile
            ),

            capability_catalog=(
                capabilities
            ),

            messages=(
                _messages(
                    agent,
                    normalized_context,
                )
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                normalized_context
            ),

            max_new_tokens=256,
        )
    )

    assert (
        provenance.task_instructions_sha256
        == comparison.task_instructions_sha256
    )


def test_execution_provenance_without_task_context(
    tmp_path: Path,
):

    clear_runtime_model_fingerprint_cache()

    model_path = (
        _write_model(
            tmp_path
            / "model"
        )
    )

    agent = (
        _agent()
    )

    profile = (
        _profile(
            model_path
        )
    )

    capabilities = (
        build_agent_capability_catalog(
            agent,
            include_arguments=True,
        )
    )

    messages = [
        {
            "role":
                "system",

            "content":
                build_worker_system_prompt(
                    agent
                ),
        },

        {
            "role":
                "user",

            "content":
                "Is jdoe locked?",
        },
    ]

    provenance = (
        build_specialist_execution_provenance(
            agent=(
                agent
            ),

            model_profile=(
                profile
            ),

            capability_catalog=(
                capabilities
            ),

            messages=(
                messages
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=None,

            max_new_tokens=256,
        )
    )

    assert (
        provenance.task_instructions_sha256
        is None
    )


def test_execution_provenance_requires_system_message(
    tmp_path: Path,
):

    clear_runtime_model_fingerprint_cache()

    model_path = (
        _write_model(
            tmp_path
            / "model"
        )
    )

    agent = (
        _agent()
    )

    with pytest.raises(
        ValueError,
        match=(
            "system message"
        ),
    ):

        build_specialist_execution_provenance(
            agent=(
                agent
            ),

            model_profile=(
                _profile(
                    model_path
                )
            ),

            capability_catalog=[
                {
                    "name":
                        "account_status",
                }
            ],

            messages=[
                {
                    "role":
                        "user",

                    "content":
                        "Is jdoe locked?",
                }
            ],

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=None,

            max_new_tokens=256,
        )


def test_execution_provenance_requires_local_model_path(
):

    agent = (
        _agent()
    )

    profile = (
        ModelProfileSettings(
            backend=(
                "qwen-funccall"
            ),

            model_path=None,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "model_path"
        ),
    ):

        build_specialist_execution_provenance(
            agent=(
                agent
            ),

            model_profile=(
                profile
            ),

            capability_catalog=[
                {
                    "name":
                        "account_status",
                }
            ],

            messages=[
                {
                    "role":
                        "system",

                    "content":
                        "SYSTEM",
                }
            ],

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=None,

            max_new_tokens=256,
        )


def test_execution_provenance_schema_alias(
    tmp_path: Path,
):

    clear_runtime_model_fingerprint_cache()

    model_path = (
        _write_model(
            tmp_path
            / "model"
        )
    )

    agent = (
        _agent()
    )

    capabilities = (
        build_agent_capability_catalog(
            agent,
            include_arguments=True,
        )
    )

    task_context = (
        "Check the account state for jdoe."
    )

    provenance = (
        build_specialist_execution_provenance(
            agent=(
                agent
            ),

            model_profile=(
                _profile(
                    model_path
                )
            ),

            capability_catalog=(
                capabilities
            ),

            messages=(
                _messages(
                    agent,
                    task_context,
                )
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            task_instructions=(
                task_context
            ),

            max_new_tokens=256,
        )
    )

    payload = (
        provenance.model_dump(
            mode="json",
            by_alias=True,
        )
    )

    assert (
        payload[
            "schema"
        ]
        == (
            "specialist-execution-provenance.v1"
        )
    )