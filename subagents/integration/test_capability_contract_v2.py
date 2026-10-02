from subagents.core.tooling.capabilities import build_capability_spec


def test_capability_contract_v2_exposes_semantics_without_authority():
    def lookup(name):
        assert name == "demo_read"
        return {
            "description": "Read one exact object.",
            "risk": "read",
            "requires_approval": False,
            "grounded_arguments": ["object_id"],
            "required_arguments": ["object_id"],
            "condition_fields": ["exists"],
            "parameters": {
                "object_id": {
                    "type": "str",
                    "description": "Exact object identifier.",
                }
            },
        }

    spec = build_capability_spec(
        "demo_read",
        tool_lookup=lookup,
        include_arguments=True,
    )

    assert spec["schema"] == "model-capability.v2"
    assert spec["contract_version"] == 2
    assert spec["effect"] == "read"
    assert spec["risk"] == "read"
    assert spec["requires_approval"] is False
    assert spec["grounded_arguments"] == ["object_id"]
    assert spec["required_arguments"] == ["object_id"]
    assert spec["condition_fields"] == ["exists"]
    assert spec["result_contract"] == {
        "type": "object",
        "condition_fields": ["exists"],
    }
    assert spec["argument_schema"]["object_id"]["type"] == "str"
