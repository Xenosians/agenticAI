import asyncio
import json

from pathlib import (
    Path,
)

import subagents.core.runtime as runtime_module

from config import (
    ModelProfileSettings,
)

from learning.execution_provenance import (
    SpecialistExecutionProvenance,
    build_specialist_execution_provenance,
    clear_runtime_model_fingerprint_cache,
)

from learning.recorder import (
    TrajectoryRecorder,
)

from learning.types import (
    TrajectoryStep,
)

from subagents.core.orchestrator import (
    Orchestrator,
)

from subagents.core.runtime import (
    AgentRuntime,
    SPECIALIST_MAX_NEW_TOKENS,
)

from subagents.core.types import (
    AgentDefinition,
    AgentResult,
    AgentTask,
    HubResult,
    SpecialistRequest,
)


def _write_model(
    root: Path,
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
        "{}",
        encoding="utf-8",
    )

    (
        root
        / "tokenizer_config.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    (
        root
        / "tokenizer.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    (
        root
        / "chat_template.jinja"
    ).write_text(
        "{{ messages }}",
        encoding="utf-8",
    )

    (
        root
        / "model.safetensors"
    ).write_bytes(
        b"fake-model-weights"
    )

    return root


def _agent(
) -> AgentDefinition:

    return (
        AgentDefinition(
            name=(
                "account-specialist"
            ),

            description=(
                "Account specialist."
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
                "auto"
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


def _capability_catalog(
) -> list[
    dict
]:

    return [
        {
            "name":
                "account_status",

            "description":
                "Check account status.",

            "argument_schema": {
                "user_id": {
                    "type":
                        "str",
                }
            },
        }
    ]


class FakeRegistry:
    def __init__(
        self,
        agent: AgentDefinition,
    ) -> None:

        self.agent = (
            agent
        )

    def get(
        self,
        name: str,
    ) -> AgentDefinition:

        if (
            name
            != self.agent.name
        ):

            raise KeyError(
                name
            )

        return (
            self.agent
        )


class FakeInference:
    def __init__(
        self,
    ) -> None:

        self.messages = (
            None
        )

        self.model_key = (
            None
        )

        self.max_new_tokens = (
            None
        )

    async def generate(
        self,
        *,
        model_key,
        messages,
        max_new_tokens,
        priority,
    ) -> str:

        self.model_key = (
            model_key
        )

        self.messages = (
            messages
        )

        self.max_new_tokens = (
            max_new_tokens
        )

        # Deliberately invalid so the runtime exits before any
        # trusted tool execution is required.
        return (
            "not-json"
        )


class NeverCalledGateway:
    async def execute(
        self,
        **kwargs,
    ):

        raise AssertionError(
            "Tool gateway must not be called."
        )


def test_runtime_captures_exact_pre_generation_provenance(
    tmp_path: Path,
    monkeypatch,
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

    exact_catalog = (
        _capability_catalog()
    )

    observed: dict = {}

    def fake_build_catalog(
        supplied_agent,
        *,
        include_arguments,
    ):

        assert (
            supplied_agent
            is agent
        )

        assert (
            include_arguments
            is True
        )

        return (
            exact_catalog
        )

    def fake_build_prompt(
        supplied_agent,
        *,
        capability_catalog,
    ):

        observed[
            "prompt_catalog"
        ] = (
            capability_catalog
        )

        assert (
            supplied_agent
            is agent
        )

        return (
            "EXACT RUNTIME SYSTEM PROMPT"
        )

    def provenance_wrapper(
        **kwargs,
    ):

        observed[
            "provenance_catalog"
        ] = (
            kwargs[
                "capability_catalog"
            ]
        )

        return (
            build_specialist_execution_provenance(
                **kwargs
            )
        )

    monkeypatch.setattr(
        runtime_module,
        "build_agent_capability_catalog",
        fake_build_catalog,
    )

    monkeypatch.setattr(
        runtime_module,
        "build_worker_system_prompt",
        fake_build_prompt,
    )

    monkeypatch.setattr(
        runtime_module,
        "build_specialist_execution_provenance",
        provenance_wrapper,
    )

    inference = (
        FakeInference()
    )

    runtime = (
        AgentRuntime(
            agent_registry=(
                FakeRegistry(
                    agent
                )
            ),

            inference=(
                inference
            ),

            tool_gateway=(
                NeverCalledGateway()
            ),

            model_profile_resolver=(
                lambda model_key: (
                    profile
                )
            ),
        )
    )

    task = (
        AgentTask(
            task_id=(
                "task-1"
            ),

            agent_name=(
                "account-specialist"
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            instructions=(
                "Check the account state for jdoe."
            ),
        )
    )

    result = (
        asyncio.run(
            runtime.run(
                task
            )
        )
    )

    assert (
        result.status
        == "error"
    )

    assert (
        result.outcome_code
        == "tool_parse_error"
    )

    # The same exact Python object was consumed by prompt
    # construction and provenance construction.
    assert (
        observed[
            "prompt_catalog"
        ]
        is exact_catalog
    )

    assert (
        observed[
            "provenance_catalog"
        ]
        is exact_catalog
    )

    assert (
        task.execution_provenance
        is not None
    )

    provenance = (
        SpecialistExecutionProvenance
        .model_validate(
            task.execution_provenance
        )
    )

    assert (
        provenance.provenance_complete
        is True
    )

    assert (
        provenance.agent_name
        == "account-specialist"
    )

    assert (
        provenance.model_key
        == "qwen2.5-0.5b-funccall"
    )

    assert (
        provenance.max_new_tokens
        == SPECIALIST_MAX_NEW_TOKENS
    )

    assert (
        inference.max_new_tokens
        == SPECIALIST_MAX_NEW_TOKENS
    )

    assert (
        inference.messages[
            0
        ][
            "content"
        ]
        == "EXACT RUNTIME SYSTEM PROMPT"
    )


def test_provenance_failure_does_not_change_runtime_result(
    monkeypatch,
):

    agent = (
        _agent()
    )

    exact_catalog = (
        _capability_catalog()
    )

    monkeypatch.setattr(
        runtime_module,
        "build_agent_capability_catalog",
        lambda *args, **kwargs: (
            exact_catalog
        ),
    )

    monkeypatch.setattr(
        runtime_module,
        "build_worker_system_prompt",
        lambda *args, **kwargs: (
            "SYSTEM"
        ),
    )

    def fail_provenance(
        **kwargs,
    ):

        raise RuntimeError(
            "synthetic provenance failure"
        )

    monkeypatch.setattr(
        runtime_module,
        "build_specialist_execution_provenance",
        fail_provenance,
    )

    inference = (
        FakeInference()
    )

    runtime = (
        AgentRuntime(
            agent_registry=(
                FakeRegistry(
                    agent
                )
            ),

            inference=(
                inference
            ),

            tool_gateway=(
                NeverCalledGateway()
            ),

            model_profile_resolver=(
                lambda model_key: (
                    ModelProfileSettings(
                        backend=(
                            "qwen-funccall"
                        ),

                        model_path=(
                            Path(
                                "/does/not/matter"
                            )
                        ),
                    )
                )
            ),
        )
    )

    task = (
        AgentTask(
            task_id="task-1",

            agent_name=(
                "account-specialist"
            ),

            user_request=(
                "Is jdoe locked?"
            ),
        )
    )

    result = (
        asyncio.run(
            runtime.run(
                task
            )
        )
    )

    assert (
        result.outcome_code
        == "tool_parse_error"
    )

    assert (
        task.execution_provenance
        is None
    )


class FakeRouter:
    async def route(
        self,
        user_request: str,
    ):

        return [
            SpecialistRequest(
                agent_name=(
                    "account-specialist"
                ),

                instructions=(
                    "Check the account state for jdoe."
                ),
            )
        ]


class ProvenanceRuntime:
    async def run(
        self,
        task,
    ) -> AgentResult:

        task.execution_provenance = {
            "marker":
                "runtime-provenance",
        }

        return (
            AgentResult(
                task_id=(
                    task.task_id
                ),

                agent_name=(
                    task.agent_name
                ),

                status=(
                    "success"
                ),

                outcome_code=(
                    "success"
                ),

                answer=(
                    "done"
                ),
            )
        )


class FakePrimaryAssistant:
    async def respond(
        self,
        user_request: str,
    ) -> str:

        return (
            "unused"
        )

    async def synthesize(
        self,
        user_request: str,
        results,
    ) -> str:

        return (
            "done"
        )


def test_orchestrator_transfers_task_provenance_to_result(
):

    orchestrator = (
        Orchestrator(
            router=(
                FakeRouter()
            ),

            runtime=(
                ProvenanceRuntime()
            ),

            primary_assistant=(
                FakePrimaryAssistant()
            ),
        )
    )

    result = (
        asyncio.run(
            orchestrator.run(
                "Is jdoe locked?"
            )
        )
    )

    assert (
        result.results[
            0
        ].execution_provenance
        == {
            "marker":
                "runtime-provenance",
        }
    )

    assert (
        result.results[
            0
        ].task_instructions
        == (
            "Check the account state for jdoe."
        )
    )


def test_recorder_persists_execution_provenance(
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

    task_context = (
        "Check the account state for jdoe."
    )

    messages = [
        {
            "role":
                "system",

            "content":
                "SYSTEM",
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
                _capability_catalog()
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

    result = (
        HubResult(
            status=(
                "success"
            ),

            user_request=(
                "Is jdoe locked?"
            ),

            routes=[
                "account-specialist",
            ],

            results=[
                AgentResult(
                    task_id=(
                        "task-1"
                    ),

                    agent_name=(
                        "account-specialist"
                    ),

                    status=(
                        "success"
                    ),

                    task_instructions=(
                        task_context
                    ),

                    execution_provenance=(
                        provenance.model_dump(
                            mode="json",
                            by_alias=True,
                        )
                    ),

                    outcome_code=(
                        "success"
                    ),

                    proposed_tool=(
                        "account_status"
                    ),

                    proposed_arguments={
                        "user_id":
                            "jdoe",
                    },

                    tool_result={
                        "ok":
                            True,
                    },

                    answer=(
                        "done"
                    ),
                )
            ],

            answer=(
                "done"
            ),
        )
    )

    output = (
        tmp_path
        / "trajectories.jsonl"
    )

    recorder = (
        TrajectoryRecorder(
            path=(
                output
            ),

            enabled=True,

            hub_model=(
                "hub-main"
            ),
        )
    )

    payload = (
        recorder.record(
            job_id=(
                "job-1"
            ),

            attempt=1,

            result=(
                result
            ),
        )
    )

    assert (
        payload
        is not None
    )

    stored = (
        json.loads(
            output.read_text(
                encoding="utf-8"
            )
            .splitlines()[
                0
            ]
        )
    )

    stored_provenance = (
        stored[
            "steps"
        ][
            0
        ][
            "execution_provenance"
        ]
    )

    assert (
        stored_provenance[
            "schema"
        ]
        == (
            "specialist-execution-provenance.v1"
        )
    )

    assert (
        stored_provenance[
            "provenance_complete"
        ]
        is True
    )

    assert (
        stored_provenance[
            "messages_sha256"
        ]
        == provenance.messages_sha256
    )

    assert (
        stored_provenance[
            "model_artifact_sha256"
        ]
        == provenance.model_artifact_sha256
    )


def test_legacy_trajectory_step_without_provenance_remains_valid(
):

    step = (
        TrajectoryStep(
            task_id=(
                "legacy-task"
            ),

            agent=(
                "account-specialist"
            ),

            status=(
                "success"
            ),
        )
    )

    assert (
        step.execution_provenance
        is None
    )