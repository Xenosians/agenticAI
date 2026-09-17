from __future__ import annotations

import json

from typing import (
    Any,
    Literal,
)

from learning.evidence.types import (
    PreferenceOption,
)


SpecialistResponseShape = Literal[
    "singular",
    "tool_calls",
]


def canonical_json(
    value: Any,
) -> str:
    """
    Deterministic JSON representation used for specialist model
    completions in DPO artifacts.

    Object keys are sorted while list order is preserved.
    """

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
    )


def normalize_tool_call(
    value: Any,
    *,
    label: str,
) -> dict[
    str,
    Any,
]:
    """
    Validate one structured worker tool call.

    This validates representation only.

    It does NOT authorize the tool.
    """

    if not isinstance(
        value,
        dict,
    ):

        raise ValueError(
            f"{label} must be a tool-call object."
        )

    name = (
        value.get(
            "name"
        )
    )

    arguments = (
        value.get(
            "arguments"
        )
    )

    if (
        not isinstance(
            name,
            str,
        )
        or not name.strip()
    ):

        raise ValueError(
            f"{label} has an invalid tool name."
        )

    if not isinstance(
        arguments,
        dict,
    ):

        raise ValueError(
            f"{label} has invalid arguments."
        )

    return {
        "name":
            name,

        "arguments":
            dict(
                arguments
            ),
    }


def normalize_tool_calls(
    value: Any,
    *,
    label: str,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """
    Validate a complete worker tool-call array.

    Runtime parsing never produces an empty validated call set, so
    an empty list is not valid preference evidence here either.
    """

    if not isinstance(
        value,
        list,
    ):

        raise ValueError(
            f"{label} must be a tool-call list."
        )

    if not value:

        raise ValueError(
            f"{label} must contain at least one tool call."
        )

    return [
        normalize_tool_call(
            call,
            label=(
                f"{label}[{index}]"
            ),
        )

        for (
            index,
            call,
        )
        in enumerate(
            value
        )
    ]


def preference_option_shape(
    option: PreferenceOption,
    *,
    label: str,
) -> SpecialistResponseShape:
    """
    Resolve the worker-response representation used by a preference
    option.

    Exactly one representation is allowed:

        singular
            tool + arguments

        tool_calls
            complete captured call array

    Mixing the two would create ambiguous training evidence.
    """

    has_tool_calls = (
        option.tool_calls
        is not None
    )

    has_tool = (
        option.tool
        is not None
    )

    has_arguments = (
        option.arguments
        is not None
    )

    if has_tool_calls:

        if (
            has_tool
            or has_arguments
        ):

            raise ValueError(
                f"{label} mixes tool_calls with "
                "singular tool/arguments."
            )

        return (
            "tool_calls"
        )

    if (
        not has_tool
        or not has_arguments
    ):

        raise ValueError(
            f"{label} is missing singular "
            "tool/arguments."
        )

    return (
        "singular"
    )


def preference_option_calls(
    option: PreferenceOption,
    *,
    label: str,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """
    Convert either supported PreferenceOption representation into
    the exact worker-call array shape.
    """

    shape = (
        preference_option_shape(
            option,
            label=(
                label
            ),
        )
    )

    if (
        shape
        == "tool_calls"
    ):

        return (
            normalize_tool_calls(
                option.tool_calls,
                label=(
                    f"{label}.tool_calls"
                ),
            )
        )

    return [
        normalize_tool_call(
            {
                "name":
                    option.tool,

                "arguments":
                    option.arguments,
            },
            label=(
                label
            ),
        )
    ]


def validate_specialist_preference_pair(
    *,
    rejected: PreferenceOption,
    chosen: PreferenceOption,
    allowed_tools: set[
        str
    ],
) -> SpecialistResponseShape:
    """
    Validate a specialist DPO pair.

    Safety / learning boundary:

        CHOSEN behavior
            must represent exactly one currently allowed tool call.

        REJECTED behavior
            is immutable negative evidence and may contain:
                - multiple calls
                - a hallucinated / unavailable tool name

    The rejected behavior is never authorization and is never
    executed by this module.

    For complete call-set corrections we require the rejected side
    to actually represent an invalid call count (>1). This prevents
    normal single-call evidence from unnecessarily switching to the
    supplemental representation.
    """

    rejected_shape = (
        preference_option_shape(
            rejected,
            label=(
                "rejected"
            ),
        )
    )

    chosen_shape = (
        preference_option_shape(
            chosen,
            label=(
                "chosen"
            ),
        )
    )

    if (
        rejected_shape
        != chosen_shape
    ):

        raise ValueError(
            "Chosen and rejected specialist responses "
            "use different representation shapes."
        )

    rejected_calls = (
        preference_option_calls(
            rejected,
            label=(
                "rejected"
            ),
        )
    )

    chosen_calls = (
        preference_option_calls(
            chosen,
            label=(
                "chosen"
            ),
        )
    )

    if (
        len(
            chosen_calls
        )
        != 1
    ):

        raise ValueError(
            "Chosen specialist response must contain "
            "exactly one tool call."
        )

    chosen_tool = (
        chosen_calls[
            0
        ][
            "name"
        ]
    )

    if (
        chosen_tool
        not in allowed_tools
    ):

        raise ValueError(
            "Chosen specialist response uses a tool "
            "that is not allowed for the target agent: "
            f"{chosen_tool}"
        )

    if (
        rejected_shape
        == "tool_calls"
        and len(
            rejected_calls
        )
        == 1
    ):

        raise ValueError(
            "tool_calls rejected evidence must represent "
            "an invalid multi-call worker response."
        )

    if (
        rejected_calls
        == chosen_calls
    ):

        raise ValueError(
            "Chosen and rejected specialist responses "
            "are identical."
        )

    return (
        rejected_shape
    )


def serialize_preference_option(
    option: PreferenceOption,
    *,
    label: str,
) -> str:
    """
    Serialize a validated preference option exactly as the worker
    model contract expects: a JSON array of tool-call objects.
    """

    calls = (
        preference_option_calls(
            option,
            label=(
                label
            ),
        )
    )

    return (
        canonical_json(
            calls
        )
    )
