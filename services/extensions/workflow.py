"""Governed workflow manifest admission.

This is preparation for n8n/native orchestration only.

Validation:
- does not execute a workflow;
- does not register a webhook;
- does not grant credentials;
- does not bypass SemanticGuard;
- does not bypass ToolGateway;
- does not convert model output into authority.

The returned plan hash can later bind durable workflow execution to the exact
validated plan.
"""

from __future__ import annotations

import hashlib
import json
import re

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


MAX_MANIFEST_BYTES = 100_000


_FORBIDDEN_ARGUMENT_KEY_PARTS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "password",
    "secret",
    "token",
}


class WorkflowNode(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    id: str = Field(
        min_length=1,
        max_length=80,
        pattern=(
            r"^[A-Za-z]"
            r"[A-Za-z0-9_-]{0,79}$"
        ),
    )

    capability: str = Field(
        min_length=1,
        max_length=150,
        pattern=(
            r"^[A-Za-z0-9_.-]+$"
        ),
    )

    arguments: dict = Field(
        default_factory=dict
    )

    depends_on: list[str] = Field(
        default_factory=list,
        max_length=20,
    )


class WorkflowManifest(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    version: int = Field(
        ge=1,
        le=1,
    )

    name: str = Field(
        min_length=1,
        max_length=120,
    )

    nodes: list[
        WorkflowNode
    ] = Field(
        min_length=1,
        max_length=20,
    )


def _contains_forbidden_secret_key(
    value,
) -> bool:
    if isinstance(
        value,
        dict,
    ):
        for key, nested in (
            value.items()
        ):
            normalized = (
                str(
                    key
                )
                .strip()
                .lower()
                .replace(
                    "-",
                    "_",
                )
            )

            if any(
                part in normalized
                for part
                in _FORBIDDEN_ARGUMENT_KEY_PARTS
            ):
                return True

            if (
                _contains_forbidden_secret_key(
                    nested
                )
            ):
                return True

    elif isinstance(
        value,
        list,
    ):
        return any(
            _contains_forbidden_secret_key(
                item
            )
            for item
            in value
        )

    return False


def _validate_dependencies(
    nodes: list[
        WorkflowNode
    ],
) -> None:
    node_ids = [
        node.id
        for node
        in nodes
    ]

    if (
        len(
            node_ids
        )
        != len(
            set(
                node_ids
            )
        )
    ):
        raise ValueError(
            "Workflow node IDs "
            "must be unique"
        )

    known = set(
        node_ids
    )

    dependencies: dict[
        str,
        set[str],
    ] = {}

    for node in nodes:
        deps = set(
            node.depends_on
        )

        if (
            len(deps)
            != len(
                node.depends_on
            )
        ):
            raise ValueError(
                "Workflow dependency "
                "list contains duplicates"
            )

        if node.id in deps:
            raise ValueError(
                "Workflow node cannot "
                "depend on itself"
            )

        unknown = (
            deps
            - known
        )

        if unknown:
            raise ValueError(
                "Workflow references "
                "an unknown dependency"
            )

        dependencies[
            node.id
        ] = deps

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(
        node_id: str,
    ) -> None:
        if node_id in visited:
            return

        if node_id in visiting:
            raise ValueError(
                "Workflow dependency "
                "cycle detected"
            )

        visiting.add(
            node_id
        )

        for dependency in (
            dependencies[
                node_id
            ]
        ):
            visit(
                dependency
            )

        visiting.remove(
            node_id
        )

        visited.add(
            node_id
        )

    for node_id in node_ids:
        visit(
            node_id
        )


def _validate_argument_types(
    *,
    arguments: dict,
    parameters: dict,
    allowed: set[str],
    required: set[str],
) -> None:
    if (
        set(
            arguments
        )
        - allowed
        or required
        - set(
            arguments
        )
    ):
        raise ValueError(
            "Workflow arguments do not "
            "match the trusted capability schema"
        )

    if (
        _contains_forbidden_secret_key(
            arguments
        )
    ):
        raise ValueError(
            "Workflow manifests cannot "
            "contain credentials or secrets"
        )

    accepted_types = {
        "str":
            str,

        "string":
            str,

        "int":
            int,

        "integer":
            int,

        "bool":
            bool,

        "boolean":
            bool,

        "list[str]":
            list,
    }

    for key, value in (
        arguments.items()
    ):
        parameter = (
            parameters[
                key
            ]
        )

        type_name = (
            parameter[
                "type"
            ]
        )

        accepted = (
            accepted_types.get(
                type_name
            )
        )

        if (
            accepted is None
            or type(
                value
            )
            is not accepted
        ):
            raise ValueError(
                "Workflow argument type "
                "is unsupported or invalid"
            )

        if (
            type_name
            == "list[str]"
            and not all(
                isinstance(
                    item,
                    str,
                )
                for item
                in value
            )
        ):
            raise ValueError(
                "Workflow string list "
                "contains an invalid value"
            )


def admit_workflow(
    value: dict,
    capabilities: dict,
) -> dict:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ).encode(
            "utf-8"
        )

    except (
        TypeError,
        ValueError,
    ):
        raise ValueError(
            "Workflow manifest "
            "is not valid JSON data"
        ) from None

    if (
        len(
            encoded
        )
        > MAX_MANIFEST_BYTES
    ):
        raise ValueError(
            "Workflow manifest "
            "exceeds size limit"
        )

    manifest = (
        WorkflowManifest
        .model_validate(
            value
        )
    )

    _validate_dependencies(
        manifest.nodes
    )

    admitted: list[
        dict
    ] = []

    from tools.capability_contract import (
        build_capability_contract,
    )

    for node in (
        manifest.nodes
    ):
        tool = (
            capabilities.get(
                node.capability
            )
        )

        if not tool:
            raise ValueError(
                "Workflow references "
                "an unregistered capability"
            )

        contract = (
            build_capability_contract(
                node.capability,
                tool,
            )
        )

        parameters = (
            tool.get(
                "parameters",
                {},
            )
        )

        allowed = set(
            contract.parameters
        )

        required = set(
            contract.required_arguments
        )

        _validate_argument_types(
            arguments=(
                node.arguments
            ),
            parameters=parameters,
            allowed=allowed,
            required=required,
        )

        effect = (
            tool.get(
                "effect"
            )
        )

        permission = (
            tool.get(
                "permission"
            )
        )

        if (
            effect
            not in {
                "read",
                "mutation",
            }
            or not permission
        ):
            raise ValueError(
                "Capability lacks "
                "explicit policy metadata"
            )

        admitted.append(
            {
                **node.model_dump(),

                # These are trusted server-derived values.
                "permission":
                    permission,

                "effect":
                    effect,

                "approval_required":
                    (
                        effect
                        == "mutation"
                    ),
            }
        )

    canonical_plan = {
        "schema":
            "governed-workflow.v1",

        "name":
            manifest.name,

        "nodes":
            admitted,
    }

    plan_sha256 = (
        hashlib.sha256(
            json.dumps(
                canonical_plan,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
            ).encode(
                "utf-8"
            )
        )
        .hexdigest()
    )

    return {
        **canonical_plan,

        "plan_sha256":
            plan_sha256,

        "execution_enabled":
            False,

        "state":
            "validated_plan",

        "required_execution_boundary":
            (
                "authenticated_service_identity "
                "-> SemanticGuard "
                "-> ToolGateway "
                "-> approval "
                "-> provider"
            ),

        "callback_contract":
            "ai-job-completion.v2",

        "retry_unknown_outcome":
            False,
    }
