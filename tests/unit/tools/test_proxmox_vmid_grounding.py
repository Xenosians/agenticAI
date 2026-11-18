"""A guest's VMID is an integer argument that must still be grounded in the request."""
from subagents.core.definitions.types import AgentDefinition
from subagents.core.orchestration.intent_contract import parse_semantic_intent
from subagents.core.orchestration.semantic_guard import SemanticGuard
from subagents.core.orchestration.semantic_binding import bind_authoritative_grounded_arguments
from subagents.core.tooling.gateway import validate_grounded_arguments
from tools.registry import get_tool

AGENT = AgentDefinition(
    name="proxmox-specialist",
    description="Proxmox VE operations.",
    tools=["proxmox_guest_status", "proxmox_guest_config"],
    model="hub-main",
    max_steps=3,
    system_prompt="",
)
REQUEST = "Read current status for VMID 990 on node hummingbird."


def intent(user=REQUEST, vmid="990"):
    parsed = parse_semantic_intent({
        "summary": user, "effect": "read",
        "allowed_tools": ["proxmox_guest_status"],
        "allowed_arguments": {"node": ["hummingbird"], "vmid": [vmid]},
        "forbidden_tools": [], "forbidden_arguments": {},
        "max_tool_calls": 1, "clarification_required": False,
    }, agent=AGENT, user_request=user)
    assert parsed is not None
    return parsed


def test_an_integer_vmid_named_in_the_request_is_grounded():
    valid, error = validate_grounded_arguments(
        user_input=REQUEST, arguments={"node": "hummingbird", "vmid": 990}, grounded_arguments=["node", "vmid"])
    assert (valid, error) == (True, None)


def test_an_integer_vmid_absent_from_the_request_is_refused():
    valid, error = validate_grounded_arguments(
        user_input=REQUEST, arguments={"node": "hummingbird", "vmid": 99}, grounded_arguments=["node", "vmid"])
    assert valid is False and "99" in str(error)
    valid, _ = validate_grounded_arguments(
        user_input="Read current status for VMID 9900 on node hummingbird.",
        arguments={"node": "hummingbird", "vmid": 990}, grounded_arguments=["node", "vmid"])
    assert valid is False


def test_a_boolean_is_not_an_integer_target():
    valid, _ = validate_grounded_arguments(
        user_input="Read VMID 1 on node hummingbird.",
        arguments={"node": "hummingbird", "vmid": True}, grounded_arguments=["node", "vmid"])
    assert valid is False


def test_semantic_guard_accepts_the_bound_integer_vmid_and_refuses_another():
    guard = SemanticGuard()
    allowed = guard.evaluate(agent=AGENT, intent=intent(), tool_name="proxmox_guest_status",
                             arguments={"node": "hummingbird", "vmid": 990})
    assert allowed.allowed, allowed.decision_code
    refused = guard.evaluate(agent=AGENT, intent=intent(), tool_name="proxmox_guest_status",
                             arguments={"node": "hummingbird", "vmid": 991})
    assert not refused.allowed


def test_binding_restores_the_bound_vmid_as_the_tool_integer():
    result = bind_authoritative_grounded_arguments(
        intent=intent(), tool_name="proxmox_guest_status", arguments={"node": "hummingbird", "vmid": "990"})
    assert result.arguments["vmid"] == 990
    assert type(result.arguments["vmid"]) is int
    tool = get_tool("proxmox_guest_status")
    assert tool is not None and tool["parameters"]["vmid"]["type"] == "int"


def test_optional_new_guest_vmids_are_grounded_when_supplied_and_omitted_when_unspecified():
    targets = {
        "proxmox_guest_create": "vmid",
        "proxmox_guest_clone": "newid",
        "proxmox_backup_restore": "newid",
    }
    for tool_name, argument in targets.items():
        tool = get_tool(tool_name)
        assert tool is not None
        assert argument in tool["grounded_arguments"]
        assert argument not in tool["required_arguments"]

        # Omission lets the trusted Proxmox policy allocate its next free ID.
        assert validate_grounded_arguments(
            user_input="Create a new guest using the requested settings.",
            arguments={},
            grounded_arguments=tool["grounded_arguments"],
        ) == (True, None)

        # A model-supplied ID is accepted only when the owner named that exact ID.
        assert validate_grounded_arguments(
            user_input="Create the new guest with VMID 992.",
            arguments={argument: 992},
            grounded_arguments=tool["grounded_arguments"],
        ) == (True, None)
        valid, error = validate_grounded_arguments(
            user_input="Create a new guest using the requested settings.",
            arguments={argument: 992},
            grounded_arguments=tool["grounded_arguments"],
        )
        assert valid is False and argument in str(error)
