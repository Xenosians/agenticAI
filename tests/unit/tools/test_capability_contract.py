import pytest

from tools.capability_contract import build_capability_contract
from tools.runtime_schema import register_tool_callable


def test_requiredness_comes_from_executable_signature():
    def example(target: str, limit: int = 10):
        return None

    register_tool_callable(example)

    contract = build_capability_contract(
        "example",
        {
            "risk": "read",
            "requires_approval": False,
            "operation_kind": "read",
            "resource_type": "example",
            "effect": "read",
            "permission": "example.read",
            "grounded_arguments": ["target"],
            "parameters": {
                "target": {"type": "str"},
                "limit": {"type": "int"},
            },
        },
    )

    assert contract.required_arguments == ("target",)
    assert contract.optional_arguments == ("limit",)
    assert contract.requiredness_source == "executable_signature"


def test_catalog_requiredness_cannot_disagree_with_runtime_signature():
    def mismatch(target: str, limit: int = 10):
        return None

    register_tool_callable(mismatch)

    with pytest.raises(ValueError, match="disagree"):
        build_capability_contract(
            "mismatch",
            {
                "risk": "read",
                "requires_approval": False,
                "operation_kind": "read",
                "resource_type": "example",
                "effect": "read",
                "permission": "example.read",
                "required_arguments": ["target", "limit"],
                "grounded_arguments": ["target"],
                "parameters": {
                    "target": {"type": "str"},
                    "limit": {"type": "int"},
                },
            },
        )


def test_semantic_roles_are_independent_from_requiredness():
    def create(
        project: str,
        summary: str,
        ticket_type: str | None = None,
    ):
        return None

    register_tool_callable(create)

    contract = build_capability_contract(
        "create",
        {
            "risk": "medium",
            "requires_approval": True,
            "operation_kind": "create",
            "resource_type": "ticket",
            "effect": "mutation",
            "permission": "ticket.create",
            "grounded_arguments": [
                "project",
                "ticket_type",
            ],
            "derived_arguments": ["summary"],
            "parameters": {
                "project": {"type": "str"},
                "summary": {"type": "str"},
                "ticket_type": {"type": "str"},
            },
        },
    )

    assert contract.required_arguments == (
        "project",
        "summary",
    )
    assert contract.optional_arguments == ("ticket_type",)
    assert contract.grounded_arguments == (
        "project",
        "ticket_type",
    )
    assert contract.derived_arguments == ("summary",)
    assert contract.effect == "mutation"
    assert contract.permission == "ticket.create"


def test_read_effect_cannot_use_mutating_crud_kind():
    def reader(target: str):
        return None

    register_tool_callable(reader)

    with pytest.raises(ValueError, match="read effect"):
        build_capability_contract(
            "reader",
            {
                "risk": "read",
                "requires_approval": False,
                "operation_kind": "update",
                "resource_type": "example",
                "effect": "read",
                "permission": "example.update",
                "parameters": {
                    "target": {"type": "str"},
                },
            },
        )


def test_trusted_policy_arguments_are_execution_only_not_model_parameters():
    def account_like_create(
        given_name: str,
        family_name: str,
        department: str | None = None,
        expected_username: str = "",
        identity_policy_version: str = "identity.v1",
    ):
        return None

    register_tool_callable(account_like_create)

    contract = build_capability_contract(
        "account_like_create",
        {
            "risk": "high",
            "requires_approval": True,
            "operation_kind": "create",
            "resource_type": "account",
            "effect": "mutation",
            "permission": "account.create",
            "grounded_arguments": [
                "given_name",
                "family_name",
                "department",
            ],
            "trusted_policy_arguments": [
                "expected_username",
                "identity_policy_version",
            ],
            "parameters": {
                "given_name": {"type": "str"},
                "family_name": {"type": "str"},
                "department": {"type": "str"},
            },
        },
    )

    assert contract.parameters == (
        "given_name",
        "family_name",
        "department",
    )

    assert contract.execution_parameters == (
        "given_name",
        "family_name",
        "department",
        "expected_username",
        "identity_policy_version",
    )

    assert contract.required_arguments == (
        "given_name",
        "family_name",
    )

    assert contract.optional_arguments == ("department",)

    assert contract.trusted_policy_arguments == (
        "expected_username",
        "identity_policy_version",
    )


def test_trusted_policy_argument_must_exist_in_executable_signature():
    def bad_policy_shape(target: str):
        return None

    register_tool_callable(bad_policy_shape)

    with pytest.raises(
        ValueError,
        match="catalog/runtime signature mismatch",
    ):
        build_capability_contract(
            "bad_policy_shape",
            {
                "risk": "medium",
                "requires_approval": True,
                "operation_kind": "update",
                "resource_type": "example",
                "effect": "mutation",
                "permission": "example.update",
                "grounded_arguments": ["target"],
                "trusted_policy_arguments": [
                    "policy_value",
                ],
                "parameters": {
                    "target": {"type": "str"},
                },
            },
        )
