from pathlib import (
    Path,
)

import yaml

from subagents.core.types import (
    AgentDefinition,
)


def load_agent_definition(
    path: str | Path,
) -> AgentDefinition:
    """
    Load an AgentDefinition from Markdown with YAML front matter.

    Every specialist must explicitly declare the logical model
    profile it uses.

    There is intentionally no implicit fallback model.
    """

    path = (
        Path(
            path
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            "Agent definition file "
            f"not found: {path}"
        )

    content = (
        path.read_text(
            encoding="utf-8"
        )
    )

    if not content.startswith(
        "---"
    ):
        raise ValueError(
            "Agent definition file must "
            "start with YAML front matter: "
            f"{path}"
        )

    parts = (
        content.split(
            "---",
            2,
        )
    )

    if (
        len(
            parts
        )
        < 3
    ):
        raise ValueError(
            f"Agent definition '{path}' "
            "has invalid YAML front matter."
        )

    metadata_text = (
        parts[1]
    )

    system_prompt = (
        parts[2]
        .strip()
    )

    metadata = (
        yaml.safe_load(
            metadata_text
        )
    )

    if not isinstance(
        metadata,
        dict,
    ):
        raise ValueError(
            f"Agent definition '{path}' "
            "contains invalid metadata."
        )

    required_fields = [
        "name",
        "description",
        "model",
    ]

    for field_name in (
        required_fields
    ):
        if (
            field_name
            not in metadata
        ):
            raise ValueError(
                f"Agent definition '{path}' "
                "is missing required field: "
                f"{field_name}"
            )

    name = (
        metadata[
            "name"
        ]
    )

    description = (
        metadata[
            "description"
        ]
    )

    model = (
        metadata[
            "model"
        ]
    )

    if (
        not isinstance(
            name,
            str,
        )
        or not name.strip()
    ):
        raise ValueError(
            f"Agent definition '{path}' "
            "has an invalid name."
        )

    if (
        not isinstance(
            description,
            str,
        )
        or not description.strip()
    ):
        raise ValueError(
            f"Agent definition '{path}' "
            "has an invalid description."
        )

    if (
        not isinstance(
            model,
            str,
        )
        or not model.strip()
    ):
        raise ValueError(
            f"Agent definition '{path}' "
            "has an invalid model."
        )

    tools = (
        metadata.get(
            "tools",
            [],
        )
    )

    if (
        not isinstance(
            tools,
            list,
        )
        or not all(
            isinstance(
                tool_name,
                str,
            )
            and bool(
                tool_name.strip()
            )

            for tool_name
            in tools
        )
    ):
        raise ValueError(
            f"Agent definition '{path}' "
            "has an invalid tools list."
        )

    max_steps = (
        metadata.get(
            "max_steps",
            3,
        )
    )

    if (
        not isinstance(
            max_steps,
            int,
        )
        or isinstance(
            max_steps,
            bool,
        )
        or max_steps < 1
    ):
        raise ValueError(
            f"Agent definition '{path}' "
            "has an invalid max_steps."
        )

    return AgentDefinition(
        name=(
            name.strip()
        ),
        description=(
            description.strip()
        ),
        model=(
            model.strip()
        ),
        tools=[
            tool_name.strip()

            for tool_name
            in tools
        ],
        max_steps=(
            max_steps
        ),
        system_prompt=(
            system_prompt
        ),
    )


def load_agent_directory(
    directory: str | Path,
) -> list[
    AgentDefinition
]:
    """
    Load every .md specialist definition from a directory.
    """

    directory = (
        Path(
            directory
        )
    )

    if not directory.exists():
        raise FileNotFoundError(
            "Agent definition directory "
            f"not found: {directory}"
        )

    if not directory.is_dir():
        raise ValueError(
            "Agent definition path is "
            "not a directory: "
            f"{directory}"
        )

    agents = []

    for path in sorted(
        directory.glob(
            "*.md"
        )
    ):
        agents.append(
            load_agent_definition(
                path
            )
        )

    return agents