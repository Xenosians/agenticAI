from __future__ import annotations

import inspect
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

from mcp.server import MCPServer


@dataclass(frozen=True)
class RuntimeToolSchema:
    name: str
    parameters: tuple[str, ...]
    required_arguments: tuple[str, ...]
    optional_arguments: tuple[str, ...]


_LOCK = RLock()
_SCHEMAS: dict[str, RuntimeToolSchema] = {}


def _schema_from_callable(fn: Callable[..., Any]) -> RuntimeToolSchema:
    signature = inspect.signature(fn)
    parameters: list[str] = []
    required: list[str] = []
    optional: list[str] = []

    for name, parameter in signature.parameters.items():
        if name in {"self", "cls"}:
            continue

        parameters.append(name)

        if parameter.default is inspect.Signature.empty:
            required.append(name)
        else:
            optional.append(name)

    return RuntimeToolSchema(
        name=fn.__name__,
        parameters=tuple(parameters),
        required_arguments=tuple(required),
        optional_arguments=tuple(optional),
    )


def register_tool_callable(fn: Callable[..., Any]) -> RuntimeToolSchema:
    schema = _schema_from_callable(fn)

    with _LOCK:
        previous = _SCHEMAS.get(schema.name)
        if previous is not None and previous != schema:
            raise ValueError(
                f"MCP capability '{schema.name}' was registered with conflicting signatures."
            )
        _SCHEMAS[schema.name] = schema

    return schema


def runtime_tool_schema(name: str) -> RuntimeToolSchema | None:
    with _LOCK:
        return _SCHEMAS.get(name)


def runtime_required_arguments(name: str) -> list[str] | None:
    schema = runtime_tool_schema(name)
    if schema is None:
        return None
    return list(schema.required_arguments)


def runtime_optional_arguments(name: str) -> list[str] | None:
    schema = runtime_tool_schema(name)
    if schema is None:
        return None
    return list(schema.optional_arguments)


def runtime_registered_tools() -> list[str]:
    with _LOCK:
        return sorted(_SCHEMAS)


def clear_runtime_schemas() -> None:
    """
    Clear captured executable schemas.

    Intended for isolated contract tests and deterministic registry
    reconstruction. Production capability declarations remain owned by
    create_mcp_server().
    """

    with _LOCK:
        _SCHEMAS.clear()


class ContractAwareMCPServer(MCPServer):
    """
    MCPServer wrapper that captures executable Python signatures.

    Requiredness comes from the real callable contract (default/no default),
    not from the language model and not from per-tool orchestration branches.
    """

    def tool(self, *args, **kwargs):
        parent_decorator = super().tool(*args, **kwargs)

        def decorator(fn):
            register_tool_callable(fn)
            return parent_decorator(fn)

        return decorator
