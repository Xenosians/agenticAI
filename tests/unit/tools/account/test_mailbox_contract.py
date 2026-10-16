import asyncio

from services.mailbox import MailboxResult
from subagents.core.definitions.types import AgentDefinition
from subagents.core.tooling.gateway import ToolGateway
from tools.account.catalog import ACCOUNT_TOOLS
from tools.mailbox.mcp import register_mailbox_tools
from tools.mailbox.catalog import format_mailbox_result


class FakeServer:
    def __init__(self):
        self.functions = {}

    def tool(self):
        def register(fn):
            self.functions[fn.__name__] = fn
            return fn
        return register


def test_create_requires_normal_approval_before_any_mcp_execution():
    calls, approvals = [], []
    class MCP:
        async def call_tool(self, *args):
            calls.append(args)
            raise AssertionError("Unapproved mailbox creation reached MCP")
    def approve(tool, arguments, **kwargs):
        approvals.append((tool, arguments))
        return {"id": "mailbox-approval-unit"}
    gateway = ToolGateway(approval_creator=approve, mcp=MCP())
    agent = AgentDefinition(name="account-specialist", description="Account worker", model="test",
                            tools=["mailbox_create", "mailbox_status"])
    result = asyncio.run(gateway.execute(agent, "Create mailbox alice@itsm.test.", "mailbox_create",
                                        {"email": "alice@itsm.test"}))
    assert result["status"] == "approval_required" and result["approval_id"] == "mailbox-approval-unit"
    assert calls == [] and approvals == [("mailbox_create", {"email": "alice@itsm.test"})]
    assert ACCOUNT_TOOLS["mailbox_create"]["risk"] == "high"


def test_mcp_contract_has_no_secret_host_or_command_parameters():
    import inspect
    class Service:
        def create(self, email):
            return MailboxResult(ok=False, status="denied", operation="mailbox_create", email=email)
        def status(self, email):
            return MailboxResult(ok=True, status="executed", operation="mailbox_status", email=email)
    server = FakeServer()
    register_mailbox_tools(server, Service())
    for name, function in server.functions.items():
        assert list(inspect.signature(function).parameters) == ["email"]
        assert set(ACCOUNT_TOOLS[name]["parameters"]) == {"email"}
        assert "password" not in function("alice@itsm.test").model_dump()


def test_presentation_does_not_claim_verification_without_evidence():
    result = MailboxResult(ok=True, status="executed", operation="mailbox_create", email="alice@itsm.test")
    assert "verified" not in format_mailbox_result(result.model_dump())


def test_invented_email_cannot_create_approval_or_call_provider():
    class MCP:
        async def call_tool(self, *args):
            raise AssertionError("Invented mailbox reached MCP")
    def approval(*args, **kwargs):
        raise AssertionError("Invented mailbox created approval")
    gateway = ToolGateway(approval_creator=approval, mcp=MCP())
    agent = AgentDefinition(name="account-specialist", description="Account worker", model="test",
                            tools=["mailbox_create", "mailbox_status"])
    result = asyncio.run(gateway.execute(agent, "Create a mailbox for Alice; I have not chosen an email.",
                                        "mailbox_create", {"email": "alice@itsm.test"}))
    assert result["ok"] is False
