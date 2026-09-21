from __future__ import annotations

from dataclasses import (
    dataclass,
)

from typing import (
    Any,
    Callable,
)

from subagents.core.definitions.types import (
    AgentDefinition,
    SemanticIntent,
)

from subagents.core.orchestration.intent_contract import (
    trusted_grounded_arguments,
    trusted_tool_effect,
)

from tools.registry import (
    get_tool,
)


ToolLookup = Callable[
    [
        str,
    ],
    dict
    | None,
]


@dataclass(
    frozen=True
)
class SemanticGuardDecision:
    allowed: bool

    decision_code: str

    error: (
        str
        | None
    ) = None


def _runtime_grounded_values(
    value: Any,
) -> tuple[
    list[str] | None,
    bool,
]:
    """
    Normalize a specialist-proposed grounded value.

    Returns:

        values
        is_collection

    Exact list values are supported for bounded multi-target
    operations such as explicit Git file staging.
    """

    if isinstance(
        value,
        str,
    ):

        if (
            not value
            or value
            != value.strip()
        ):

            return (
                None,
                False,
            )

        return (
            [
                value,
            ],
            False,
        )

    if isinstance(
        value,
        list,
    ):

        if not value:

            return (
                None,
                True,
            )

        values: list[str] = []

        seen: set[str] = set()

        for item in value:

            if not isinstance(
                item,
                str,
            ):

                return (
                    None,
                    True,
                )

            if (
                not item
                or item
                != item.strip()
            ):

                return (
                    None,
                    True,
                )

            if item in seen:

                return (
                    None,
                    True,
                )

            seen.add(
                item
            )

            values.append(
                item
            )

        return (
            values,
            True,
        )

    return (
        None,
        False,
    )


class SemanticGuard:

    def __init__(
        self,
        *,
        tool_lookup: ToolLookup = get_tool,
    ) -> None:

        self.tool_lookup = (
            tool_lookup
        )

    def _deny(
        self,
        decision_code: str,
        error: str,
    ) -> SemanticGuardDecision:

        return (
            SemanticGuardDecision(
                allowed=False,

                decision_code=(
                    decision_code
                ),

                error=(
                    error
                ),
            )
        )

    def _allow(
        self,
    ) -> SemanticGuardDecision:

        return (
            SemanticGuardDecision(
                allowed=True,

                decision_code=(
                    "semantic_guard_allowed"
                ),

                error=None,
            )
        )

    def evaluate(
        self,
        *,
        agent: AgentDefinition,
        intent: SemanticIntent,
        tool_name: str,
        arguments: dict,
    ) -> SemanticGuardDecision:

        # ========================================================
        # INTENT EXECUTABILITY
        # ========================================================

        if (
            intent.clarification_required
        ):

            return (
                self._deny(
                    "semantic_clarification_required",
                    (
                        "The request requires clarification "
                        "before a governed capability can be used."
                    ),
                )
            )

        if (
            intent.max_tool_calls
            != 1
        ):

            return (
                self._deny(
                    "semantic_call_cardinality_invalid",
                    (
                        "The semantic contract does not match "
                        "the current exactly-one-call runtime."
                    ),
                )
            )

        if (
            intent.effect
            == "unknown"
        ):

            return (
                self._deny(
                    "semantic_effect_unknown",
                    (
                        "The requested operation does not yet "
                        "have a safe resolved effect."
                    ),
                )
            )

        # ========================================================
        # TRUSTED AGENT BOUNDARY
        # ========================================================

        trusted_agent_tools = {
            value.strip()

            for value
            in agent.tools

            if (
                isinstance(
                    value,
                    str,
                )
                and value.strip()
            )
        }

        if (
            tool_name
            not in trusted_agent_tools
        ):

            return (
                self._deny(
                    "semantic_tool_not_available",
                    (
                        "The proposed capability is not available "
                        "to this specialist."
                    ),
                )
            )

        # ========================================================
        # SEMANTIC CAPABILITY BOUNDARY
        # ========================================================

        if (
            tool_name
            in intent.forbidden_tools
        ):

            return (
                self._deny(
                    "semantic_tool_forbidden",
                    (
                        "The proposed capability conflicts with "
                        "an explicit semantic exclusion."
                    ),
                )
            )

        if (
            tool_name
            not in intent.allowed_tools
        ):

            return (
                self._deny(
                    "semantic_tool_not_allowed",
                    (
                        "The proposed capability does not match "
                        "the resolved semantic intent."
                    ),
                )
            )

        # ========================================================
        # TRUSTED TOOL METADATA
        # ========================================================

        tool = (
            self.tool_lookup(
                tool_name
            )
        )

        if tool is None:

            return (
                self._deny(
                    "semantic_tool_metadata_missing",
                    (
                        "Trusted capability metadata could not "
                        "be resolved."
                    ),
                )
            )

        try:

            trusted_effect = (
                trusted_tool_effect(
                    tool
                )
            )

            grounded_arguments = (
                trusted_grounded_arguments(
                    tool_name=(
                        tool_name
                    ),

                    tool=(
                        tool
                    ),
                )
            )

        except ValueError as exc:

            return (
                self._deny(
                    "semantic_tool_metadata_invalid",
                    str(
                        exc
                    ),
                )
            )

        # ========================================================
        # EFFECT
        # ========================================================

        if (
            trusted_effect
            != intent.effect
        ):

            return (
                self._deny(
                    "semantic_effect_mismatch",
                    (
                        "The proposed capability effect does not "
                        "match the resolved semantic intent."
                    ),
                )
            )

        if not isinstance(
            arguments,
            dict,
        ):

            return (
                self._deny(
                    "semantic_arguments_invalid",
                    (
                        "The specialist produced invalid "
                        "structured arguments."
                    ),
                )
            )

        grounded_set = (
            set(
                grounded_arguments
            )
        )

        # ========================================================
        # EXPLICIT BINDINGS MUST BE PRESENT
        # ========================================================

        for argument_name in (
            intent.allowed_arguments
        ):

            if (
                argument_name
                not in grounded_set
            ):

                continue

            if (
                argument_name
                not in arguments
            ):

                return (
                    self._deny(
                        "semantic_bound_argument_missing",
                        (
                            "The specialist omitted an argument "
                            "that was semantically bound by the "
                            "Hub intent."
                        ),
                    )
                )

        # ========================================================
        # GROUNDED ARGUMENTS
        # ========================================================

        for argument_name in (
            grounded_arguments
        ):

            if (
                argument_name
                not in arguments
            ):

                continue

            raw_value = (
                arguments[
                    argument_name
                ]
            )

            (
                values,
                is_collection,
            ) = (
                _runtime_grounded_values(
                    raw_value
                )
            )

            if values is None:

                return (
                    self._deny(
                        "semantic_grounded_argument_invalid",
                        (
                            "A semantically grounded argument "
                            "must be a non-empty exact string "
                            "or exact list of strings."
                        ),
                    )
                )

            forbidden_values = (
                intent
                .forbidden_arguments
                .get(
                    argument_name,
                    [],
                )
            )

            if any(
                value
                in forbidden_values

                for value
                in values
            ):

                return (
                    self._deny(
                        "semantic_argument_forbidden",
                        (
                            "The proposed grounded argument "
                            "contains an explicitly forbidden "
                            "semantic target."
                        ),
                    )
                )

            allowed_values = (
                intent
                .allowed_arguments
                .get(
                    argument_name
                )
            )

            if not allowed_values:

                return (
                    self._deny(
                        "semantic_argument_unbound",
                        (
                            "The specialist proposed a grounded "
                            "argument that was not bound by the "
                            "semantic intent."
                        ),
                    )
                )

            if is_collection:

                # --------------------------------------------
                # EXACT COLLECTION SEMANTICS
                #
                # User:
                #     stage a.py and b.py
                #
                # Hub:
                #     paths = [a.py, b.py]
                #
                # Specialist must propose exactly that set.
                #
                # Neither:
                #     [a.py]
                #
                # nor:
                #     [a.py, b.py, c.py]
                #
                # is acceptable.
                # --------------------------------------------

                if (
                    set(
                        values
                    )
                    != set(
                        allowed_values
                    )
                ):

                    return (
                        self._deny(
                            "semantic_argument_not_allowed",
                            (
                                "The proposed grounded argument "
                                "collection does not exactly "
                                "match the semantic targets."
                            ),
                        )
                    )

            else:

                if (
                    values[
                        0
                    ]
                    not in allowed_values
                ):

                    return (
                        self._deny(
                            "semantic_argument_not_allowed",
                            (
                                "The proposed grounded argument "
                                "does not match the semantic target."
                            ),
                        )
                    )

        return (
            self._allow()
        )