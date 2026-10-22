import inspect
import json

from services.directory import MockDirectoryService
from tools.directory.catalog import (
    DIRECTORY_TOOLS,
    format_access_explain_result,
    format_account_search_result,
    format_group_members_result,
    format_user_groups_result,
)
from subagents.core.orchestration.candidate_selector import _short
from tools.directory.mcp import register_directory_tools
from tools.registry import TOOLS


NEW_TOOLS = {"account_search", "user_groups", "group_members", "access_explain"}


class FakeMCPServer:

    def __init__(self):
        self.functions = {}

    def tool(self):
        def decorator(function):
            self.functions[function.__name__] = function
            return function
        return decorator


def registered(directory=None):
    server = FakeMCPServer()
    directory = directory or MockDirectoryService()
    register_directory_tools(server, directory)
    return server.functions, directory


def test_directory_read_tools_are_registered_with_exact_signatures():
    functions, _ = registered()
    assert set(functions) == NEW_TOOLS
    signatures = {name: list(inspect.signature(fn).parameters) for name, fn in functions.items()}
    assert signatures == {
        "account_search": ["query"],
        "user_groups": ["user_id"],
        "group_members": ["group"],
        "access_explain": ["user_id", "resource"],
    }
    for function in functions.values():
        assert all(p.default is inspect.Parameter.empty for p in inspect.signature(function).parameters.values())


def test_catalog_entries_are_grounded_reads_without_approval():
    assert set(DIRECTORY_TOOLS) == NEW_TOOLS
    for name, spec in DIRECTORY_TOOLS.items():
        assert TOOLS[name] is spec
        assert spec["risk"] == "read" and spec["requires_approval"] is False
        assert TOOLS[name]["effect"] == "read"
        assert "approval_formatter" not in spec and "policy_resolver" not in spec
        assert set(spec["grounded_arguments"]) == set(spec["parameters"])
        assert callable(spec["result_formatter"])


def test_read_only_boundary_survives_router_description_shortening():
    for spec in DIRECTORY_TOOLS.values():
        assert _short(spec["description"], 200).startswith("Read only, ")


def test_access_explain_suggests_fixes_without_executing_them():
    calls = []

    class RecordingDirectory(MockDirectoryService):
        def unlock_user(self, user_id):
            calls.append(("unlock_user", user_id))
            return super().unlock_user(user_id)

    functions, directory = registered(RecordingDirectory())
    result = functions["access_explain"](user_id="asmith", resource="vpn")
    assert result.ok is True and result.has_access is False
    assert [(r.code, r.suggested_tool) for r in result.reasons] == [("account_locked", "unlock_user")]
    assert calls == [] and directory.users["asmith"]["locked"] is True
    text = format_access_explain_result(result.model_dump())
    assert "unlock_user" in text and "Nothing was changed." in text


def test_mcp_results_are_structured_and_never_carry_credentials():
    functions, directory = registered()
    directory.users["jdoe"]["email"] = "jdoe@itsm.test"
    search = functions["account_search"](query="jdoe").model_dump()
    assert search["accounts"] == [{"username": "jdoe", "display_name": None, "email": "jdoe@itsm.test",
                                   "enabled": True, "locked": False}]
    groups = functions["user_groups"](user_id="jdoe").model_dump()
    assert groups["groups"] == [{"name": "VPN-Users", "membership": "direct", "resources": ["vpn"]}]
    members = functions["group_members"](group="vpn").model_dump()
    assert [m["name"] for m in members["members"]] == ["asmith", "jdoe"]
    for payload in (search, groups, members):
        assert "password" not in json.dumps(payload).lower()


def test_mcp_reports_provider_errors_honestly():
    functions, _ = registered()
    result = functions["account_search"](query="j")
    assert result.ok is False and result.error_code == "invalid_argument"
    missing = functions["user_groups"](user_id="nobody")
    assert missing.ok is False and missing.error_code == "not_found"
    assert format_user_groups_result(missing.model_dump()).startswith("The directory group lookup did not succeed")


def test_formatters_summarize_bounded_lists():
    accounts = [{"username": f"u{i}", "enabled": True, "locked": False} for i in range(30)]
    text = format_account_search_result({"ok": True, "query": "u", "accounts": accounts, "truncated": True})
    assert "and 5 more" in text and "capped" in text
    assert format_account_search_result({"ok": True, "query": "zz", "accounts": []}) == "No directory accounts match 'zz'."
    members = format_group_members_result({"ok": True, "group": "VPN-Users", "resources": ["vpn"], "members": [
        {"name": "jdoe", "type": "user", "membership": "direct"}]})
    assert members == "The group VPN-Users (grants vpn) has 1 member(s): jdoe (user, direct)."
    granted = format_access_explain_result({"ok": True, "user_id": "jdoe", "resource": "vpn", "group": "VPN-Users",
                                            "has_access": True, "checks": {"membership": "nested"}, "reasons": []})
    assert "nested member of VPN-Users" in granted and "outside the directory" in granted
