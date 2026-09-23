from __future__ import annotations

import json

from typing import (
    Any,
)

from subagents.core.tooling.capabilities import (
    build_agent_capability_catalog,
)

from subagents.core.definitions.types import (
    AgentDefinition,
)

from subagents.prompts.prompt_loader import (
    load_prompt,
)


def _compact_argument(
    name: str,
    schema: dict[
        str,
        Any,
    ],
) -> str:
    argument_type = (
        schema.get(
            "type"
        )
        or "value"
    )

    enum = (
        schema.get(
            "enum"
        )
    )

    if (
        isinstance(
            enum,
            list,
        )
        and enum
    ):
        encoded = (
            "|".join(
                str(
                    item
                )
                for item
                in enum
            )
        )

        return (
            f"{name}:{argument_type}="
            f"[{encoded}]"
        )

    return (
        f"{name}:{argument_type}"
    )


def build_compact_capability_text(
    capabilities: list[
        dict[
            str,
            Any,
        ]
    ],
) -> str:
    """
    Render a bounded, low-token capability view for small specialist
    models.

    This is model-facing metadata only. The trusted registry,
    SemanticGuard, ToolGateway, approval policy, and provider execution
    remain authoritative.
    """

    lines: list[str] = []

    for capability in capabilities:
        name = (
            capability.get(
                "name"
            )
        )

        if not isinstance(
            name,
            str,
        ) or not name.strip():
            raise ValueError(
                "Compact capability contains an invalid name."
            )

        schema = (
            capability.get(
                "argument_schema",
                {},
            )
        )

        if not isinstance(
            schema,
            dict,
        ):
            raise ValueError(
                f"Capability '{name}' has an invalid argument schema."
            )

        arguments = [
            _compact_argument(
                argument_name,
                argument_schema,
            )
            for (
                argument_name,
                argument_schema,
            ) in schema.items()
            if isinstance(
                argument_schema,
                dict,
            )
        ]

        lines.append(
            f"- {name}("
            + ", ".join(
                arguments
            )
            + ")"
        )

    return (
        "\n".join(
            lines
        )
    )


def build_worker_system_prompt(
    agent: AgentDefinition,
    *,
    capability_catalog: (
        list[
            dict[
                str,
                Any,
            ]
        ]
        | None
    ) = None,
    prompt_profile: str = "standard",
) -> str:
    """
    Build the worker prompt from the specialist definition and trusted
    runtime capability catalog.

    `standard` preserves the original verbose protocol.

    `compact` renders the same capability names and argument schemas in
    a bounded representation intended for small specialist models.
    Compact mode changes only model-facing prompting; it never changes
    authority, policy, approval, grounding, or execution.
    """

    capabilities = (
        capability_catalog

        if capability_catalog
        is not None

        else (
            build_agent_capability_catalog(
                agent,
                include_arguments=True,
            )
        )
    )

    normalized_profile = (
        prompt_profile
        .strip()
        .lower()
    )

    if (
        normalized_profile
        == "standard"
    ):
        capability_json = (
            json.dumps(
                capabilities,
                indent=2,
            )
        )

        template = (
            load_prompt(
                "worker_tool_protocol.txt"
            )
        )

        return (
            template
            .replace(
                "{{AGENT_SYSTEM_PROMPT}}",
                agent.system_prompt,
            )
            .replace(
                "{{TOOLS_JSON}}",
                capability_json,
            )
        )

    if (
        normalized_profile
        == "compact"
    ):
        template = (
            load_prompt(
                "worker_tool_protocol_compact.txt"
            )
        )

        return (
            template
            .replace(
                "{{AGENT_SYSTEM_PROMPT}}",
                agent.system_prompt,
            )
            .replace(
                "{{TOOLS_COMPACT}}",
                build_compact_capability_text(
                    capabilities
                ),
            )
        )

    raise ValueError(
        "Unsupported worker prompt profile: "
        f"{prompt_profile}"
    )
