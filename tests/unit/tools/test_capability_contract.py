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
            "grounded_arguments": ["target"],
            "parameters": {
                "target": {"type": "str"},
                "limit": {"type": "int"},
            },
        },
    )

    assert contract.required_arguments == ("target",)
    assert contract.optional_arguments == ("limit",)


def test_semantic_roles_are_independent_from_requiredness():
    def create(project: str, summary: str, ticket_type: str | None = None):
        return None

    register_tool_callable(create)

    contract = build_capability_contract(
        "create",
        {
            "risk": "medium",
            "requires_approval": True,
            "grounded_arguments": ["project", "ticket_type"],
            "derived_arguments": ["summary"],
            "parameters": {
                "project": {"type": "str"},
                "summary": {"type": "str"},
                "ticket_type": {"type": "str"},
            },
        },
    )

    assert contract.required_arguments == ("project", "summary")
    assert contract.optional_arguments == ("ticket_type",)
    assert contract.grounded_arguments == ("project", "ticket_type")
    assert contract.derived_arguments == ("summary",)
