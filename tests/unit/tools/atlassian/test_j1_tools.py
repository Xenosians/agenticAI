from tools.atlassian.catalog import ATLASSIAN_TOOLS
from tools.atlassian.mcp import register_atlassian_tools


class FakeServer:
    def __init__(self):
        self.functions = {}

    def tool(self):
        def decorator(function):
            self.functions[function.__name__] = function
            return function
        return decorator


class FakeService:
    def credential_status(self):
        return {
            "ok": True,
            "status": "success",
            "admin_api_key_configured": True,
            "jira_basic_configured": False,
            "default_org_configured": True,
            "admin_auth_type": "api_key",
            "jira_auth_type": None,
            "secret_values_exposed": False,
        }

    def list_organizations(self, *, limit):
        return {
            "ok": True,
            "status": "success",
            "organizations": [],
            "count": 0,
            "truncated": False,
            "next_cursor_available": False,
            "error": None,
        }

    def get_organization(self, org_id=None):
        return {
            "ok": True,
            "status": "success",
            "organization": {
                "id": org_id or "org-default",
                "name": "Example",
                "type": "orgs",
            },
            "error": None,
        }

    def list_workspaces(self, *, org_id=None, name=None, limit):
        return {
            "ok": True,
            "status": "success",
            "organization_id": org_id or "org-default",
            "workspaces": [],
            "count": 0,
            "truncated": False,
            "next_cursor_available": False,
            "error": None,
        }

    def list_api_token_metadata(self, *, account_id, limit):
        return {
            "ok": True,
            "status": "success",
            "account_id": account_id,
            "tokens": [],
            "count": 0,
            "truncated": False,
            "secret_values_exposed": False,
            "error": None,
        }


def test_all_j1_tools_are_read_only():
    assert set(ATLASSIAN_TOOLS) == {
        "atlassian_credential_status",
        "atlassian_org_list",
        "atlassian_org_get",
        "atlassian_workspace_list",
        "atlassian_api_token_metadata",
    }

    for tool in ATLASSIAN_TOOLS.values():
        assert tool["risk"] == "read"
        assert tool["requires_approval"] is False

    assert ATLASSIAN_TOOLS["atlassian_api_token_metadata"]["grounded_arguments"] == [
        "account_id"
    ]


def test_mcp_registers_all_j1_tools():
    server = FakeServer()
    register_atlassian_tools(server, FakeService())

    assert set(server.functions) == set(ATLASSIAN_TOOLS)

    credential_result = server.functions["atlassian_credential_status"]()
    assert credential_result.admin_api_key_configured is True
    assert credential_result.secret_values_exposed is False

    token_result = server.functions["atlassian_api_token_metadata"](
        account_id="account-1"
    )
    assert token_result.secret_values_exposed is False
