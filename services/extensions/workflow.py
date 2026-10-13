"""Future n8n/native workflow admission. Validation cannot execute a workflow."""
from pydantic import BaseModel, ConfigDict, Field


class WorkflowNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=80)
    capability: str = Field(min_length=1, max_length=150)
    arguments: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list, max_length=20)


class WorkflowManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = Field(ge=1, le=1)
    name: str = Field(min_length=1, max_length=120)
    nodes: list[WorkflowNode] = Field(min_length=1, max_length=20)


def admit_workflow(value: dict, capabilities: dict) -> dict:
    manifest = WorkflowManifest.model_validate(value)
    seen = set()
    admitted = []
    for node in manifest.nodes:
        if node.id in seen or not set(node.depends_on).issubset(seen):
            raise ValueError("Workflow requires unique nodes in dependency order; cycles are not accepted")
        tool = capabilities.get(node.capability)
        if not tool:
            raise ValueError("Workflow references an unregistered capability")
        from tools.capability_contract import build_capability_contract
        contract = build_capability_contract(node.capability, tool)
        parameters = tool.get("parameters", {})
        allowed = set(contract.parameters)
        required = set(contract.required_arguments)
        if set(node.arguments) - allowed or required - set(node.arguments):
            raise ValueError("Workflow arguments do not match the trusted capability schema")
        for key, value in node.arguments.items():
            type_name = parameters[key]["type"]
            accepted = {"str": str, "string": str, "int": int, "integer": int,
                        "bool": bool, "boolean": bool, "list[str]": list}.get(type_name)
            if accepted is None or type(value) is not accepted:
                raise ValueError("Workflow argument type is unsupported or invalid")
            if type_name == "list[str]" and not all(isinstance(v, str) for v in value):
                raise ValueError("Workflow string list contains an invalid value")
        if tool.get("effect") not in {"read", "mutation"} or not tool.get("permission"):
            raise ValueError("Capability lacks explicit policy metadata")
        admitted.append({**node.model_dump(), "permission": tool["permission"], "effect": tool["effect"],
                         "approval_required": tool["effect"] == "mutation"})
        seen.add(node.id)
    return {"schema": "governed-workflow.v1", "name": manifest.name, "nodes": admitted,
            "execution_enabled": False, "state": "validated_plan",
            "required_execution_boundary": "authenticated_service_identity -> SemanticGuard -> ToolGateway -> approval -> provider",
            "callback_contract": "ai-job-completion.v2", "retry_unknown_outcome": False}
