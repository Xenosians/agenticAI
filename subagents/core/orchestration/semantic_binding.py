from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from subagents.core.definitions.types import SemanticIntent
from subagents.core.orchestration.intent_contract import trusted_grounded_arguments
from tools.registry import get_tool


ToolLookup = Callable[[str], dict | None]


@dataclass(frozen=True)
class SemanticBindingEvent:
    """Audit evidence for deterministic runtime rebinding."""

    argument_name: str
    action: str
    worker_value: Any
    bound_value: Any
    source: str = "trusted_semantic_contract"


@dataclass(frozen=True)
class SemanticBindingResult:
    """Arguments after deterministic binding plus audit events."""

    arguments: dict[str, Any]
    events: tuple[SemanticBindingEvent, ...]


def _parameter_expects_collection(
    *,
    tool: dict,
    argument_name: str,
) -> bool:
    """Resolve collection shape only from trusted capability metadata."""

    parameters = tool.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(
            "Trusted capability contains invalid parameters metadata."
        )

    parameter = parameters.get(argument_name, {})
    if not isinstance(parameter, dict):
        return False

    value_type = parameter.get("type")
    if not isinstance(value_type, str):
        return False

    normalized = value_type.strip().lower().replace(" ", "")

    return (
        normalized in {"list", "array", "seq"}
        or normalized.startswith("list[")
        or normalized.startswith("array[")
        or normalized.startswith("seq[")
    )


def _bound_runtime_value(
    *,
    trusted_value: str,
    expects_collection: bool,
) -> Any:
    if expects_collection:
        return [trusted_value]
    return trusted_value


def bind_authoritative_grounded_arguments(
    *,
    intent: SemanticIntent,
    tool_name: str,
    arguments: dict[
        str,
        Any,
    ],
    tool_lookup: ToolLookup = get_tool,
) -> SemanticBindingResult:
    """
    Apply deterministic authority-preserving binding to one worker
    proposal.

    Model output is a proposal, never authorization.

    Generic rules:

    1. Only trusted capability grounded_arguments participate.

    2. No trusted semantic value:
       - invent nothing;
       - leave worker value visible;
       - SemanticGuard may reject it as unbound.

    3. Exactly one trusted semantic value:
       - if worker proposes an explicitly forbidden value, preserve it
         so SemanticGuard can reject the exclusion;
       - otherwise insert/restore the trusted semantic value.

    4. Multiple trusted semantic values:
       - do not guess;
       - leave proposal unchanged for SemanticGuard.

    5. Non-grounded / derived arguments are never rewritten here.

    This narrows model freedom. It never widens user authority.
    """

    if not isinstance(
        arguments,
        dict,
    ):

        raise ValueError(
            "Specialist arguments must be an object "
            "before semantic binding."
        )

    tool = (
        tool_lookup(
            tool_name
        )
    )

    if tool is None:

        raise ValueError(
            f"Unknown trusted capability: {tool_name}"
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

    rebound = (
        dict(
            arguments
        )
    )

    events: list[
        SemanticBindingEvent
    ] = []

    for argument_name in (
        grounded_arguments
    ):

        allowed_values = (
            intent
            .allowed_arguments
            .get(
                argument_name
            )
        )

        # --------------------------------------------------------
        # NO BINDING
        #
        # Runtime must not manufacture a semantic target.
        # --------------------------------------------------------

        if not allowed_values:

            continue

        # --------------------------------------------------------
        # AMBIGUOUS / MULTI-VALUE BINDING
        #
        # Generic code cannot safely infer whether this means:
        #
        #   - choose one target
        #   - preserve a collection
        #   - operate over multiple targets
        #
        # Leave it to the existing semantic contract / guard.
        # --------------------------------------------------------

        if (
            len(
                allowed_values
            )
            != 1
        ):

            continue

        trusted_value = (
            allowed_values[
                0
            ]
        )

        if (
            not isinstance(
                trusted_value,
                str,
            )
            or not trusted_value
            or trusted_value
            != trusted_value.strip()
        ):

            raise ValueError(
                "Semantic contract contains an invalid "
                "grounded value."
            )

        expects_collection = (
            _parameter_expects_collection(
                tool=(
                    tool
                ),

                argument_name=(
                    argument_name
                ),
            )
        )

        bound_value = (
            _bound_runtime_value(
                trusted_value=(
                    trusted_value
                ),

                expects_collection=(
                    expects_collection
                ),
            )
        )

        worker_present = (
            argument_name
            in arguments
        )

        worker_value = (
            arguments.get(
                argument_name
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

        # --------------------------------------------------------
        # EXPLICIT EXCLUSIONS HAVE PRECEDENCE OVER AUTO-REPAIR
        #
        # If the specialist actually proposes a forbidden target,
        # keep that exact proposal intact.
        #
        # SemanticGuard must see it and reject it.
        # --------------------------------------------------------

        worker_proposes_forbidden = False

        if isinstance(
            worker_value,
            str,
        ):

            worker_proposes_forbidden = (
                worker_value
                in forbidden_values
            )

        elif isinstance(
            worker_value,
            list,
        ):

            worker_proposes_forbidden = any(
                (
                    isinstance(
                        item,
                        str,
                    )
                    and item
                    in forbidden_values
                )

                for item
                in worker_value
            )

        if worker_proposes_forbidden:

            continue

        # --------------------------------------------------------
        # ALREADY CORRECT
        # --------------------------------------------------------

        if (
            worker_value
            == bound_value
        ):

            continue

        # --------------------------------------------------------
        # ORDINARY MODEL DRIFT
        #
        # There is exactly one trusted semantic target and the
        # worker did not explicitly violate an exclusion.
        #
        # Runtime owns the authority-bearing value.
        # --------------------------------------------------------

        rebound[
            argument_name
        ] = (
            bound_value
        )

        events.append(
            SemanticBindingEvent(
                argument_name=(
                    argument_name
                ),

                action=(
                    "rebound"
                    if worker_present
                    else "inserted"
                ),

                worker_value=(
                    worker_value
                ),

                bound_value=(
                    bound_value
                ),
            )
        )

    return (
        SemanticBindingResult(
            arguments=(
                rebound
            ),

            events=(
                tuple(
                    events
                )
            ),
        )
    )
