import asyncio
import inspect
from pathlib import Path

import pytest

import services.workspace_repositories as repositories
import tools.developer.execution as developer
import tools.developer.mcp as developer_mcp
from services.git_repositories import GitRepositoryRegistry
from subagents.core.definitions.types import AgentDefinition
from subagents.core.tooling.capabilities import build_capability_spec
from subagents.core.tooling.gateway import ToolGateway
from tools.developer.presentation import (
    build_developer_execution_card,
    build_project_info_card,
    format_developer_execution_approval,
    format_developer_execution_result,
    format_project_info_result,
)
from tools.registry import get_tool


PROJECT_TOOLS = (
    "workspace_project_info",
    "workspace_run_tests",
    "workspace_run_build",
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    paths = {}
    for alias in ("ai", "backend", "frontend"):
        root = tmp_path / alias
        (root / ".git").mkdir(parents=True)
        (root / "project").mkdir()
        (root / "project" / "requirements.txt").write_text("pytest\n")
        paths[alias] = root

    registry = GitRepositoryRegistry(
        root=tmp_path,
        repositories=paths,
        default_repository="ai",
    )
    monkeypatch.setattr(repositories, "build_git_repository_registry", lambda: registry)

    calls = []

    def run_process(**kwargs):
        calls.append(kwargs)
        return {
            "ok": True,
            "status": "success",
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "duration_ms": 1,
        }

    monkeypatch.setattr(developer, "run_trusted_process", run_process)
    return registry, paths, calls


@pytest.mark.parametrize("tool_name", PROJECT_TOOLS)
@pytest.mark.parametrize("alias", ("ai", "backend", "frontend"))
def test_project_tools_use_selected_canonical_repository(workspace, tool_name, alias):
    registry, paths, calls = workspace
    result = getattr(developer, tool_name)(relative_path="project", repository=alias.upper())

    assert result["ok"] is True
    assert result["repository"] == alias
    assert result.get("path", result.get("project_path")) == "project"
    if tool_name == "workspace_project_info":
        assert calls == []
    else:
        assert len(calls) == 1
        assert Path(calls[0]["cwd"]) == paths[alias] / "project"
        assert calls[0]["args"] == (
            ["-m", "pytest", "-q"] if tool_name == "workspace_run_tests"
            else ["-m", "compileall", "-q", "."]
        )


@pytest.mark.parametrize("tool_name", PROJECT_TOOLS)
def test_omitted_repository_preserves_configured_default(workspace, tool_name):
    registry, paths, calls = workspace
    # A non-ai default proves omission is configuration-driven.
    registry.default_repository = "backend"
    result = getattr(developer, tool_name)(relative_path="project")
    assert result["ok"] is True
    assert result["repository"] == "backend"
    if calls:
        assert Path(calls[0]["cwd"]) == paths["backend"] / "project"


@pytest.mark.parametrize("tool_name", PROJECT_TOOLS)
@pytest.mark.parametrize("repository", ("unknown", "../backend", "/tmp", "C:\\repo", "", 42))
def test_invalid_repository_never_reaches_execution(workspace, tool_name, repository):
    _, _, calls = workspace
    result = getattr(developer, tool_name)(relative_path="project", repository=repository)
    assert result["ok"] is False
    assert result["status"] == "denied"
    assert calls == []


@pytest.mark.parametrize("tool_name", PROJECT_TOOLS)
@pytest.mark.parametrize("relative_path", (
    "../ai/project", "project/../../ai/project", "project/../project",
    "..\\ai\\project", "/tmp", "C:\\repo", "C:repo", "\\repo", "bad\x00path",
))
def test_paths_cannot_cross_selected_repository(workspace, tool_name, relative_path):
    _, _, calls = workspace
    result = getattr(developer, tool_name)(relative_path=relative_path, repository="backend")
    assert result["ok"] is False
    assert result["status"] == "denied"
    assert calls == []


@pytest.mark.parametrize("tool_name", PROJECT_TOOLS)
def test_symlink_cannot_redirect_into_another_repository(workspace, tool_name):
    _, paths, calls = workspace
    (paths["backend"] / "linked").symlink_to(paths["ai"] / "project", target_is_directory=True)
    result = getattr(developer, tool_name)(relative_path="linked", repository="backend")
    assert result["status"] == "denied"
    assert "outside" in result["error"]
    assert calls == []


@pytest.mark.parametrize("tool_name", PROJECT_TOOLS)
def test_repository_alias_does_not_become_a_relative_path(workspace, tool_name):
    _, _, calls = workspace
    result = getattr(developer, tool_name)(relative_path="backend")
    assert result["status"] == "denied"
    assert calls == []


@pytest.mark.parametrize("tool_name", PROJECT_TOOLS)
def test_repository_enum_comes_from_the_shared_registry(workspace, tool_name):
    registry, _, _ = workspace
    registry.repositories["preview"] = registry.repositories["frontend"]
    spec = build_capability_spec(tool_name)
    assert spec["argument_schema"]["repository"]["enum"] == registry.available()
    assert "repository" in spec["grounded_arguments"]
    assert "repository" not in spec["required_arguments"]
    assert all("/" not in value and "\\" not in value
               for value in spec["argument_schema"]["repository"]["enum"])


class CapturingServer:
    def __init__(self):
        self.functions = {}

    def tool(self):
        def register(function):
            self.functions[function.__name__] = function
            return function
        return register


@pytest.mark.parametrize("tool_name", PROJECT_TOOLS)
def test_mcp_preserves_alias_and_nullable_defaults(workspace, tool_name):
    _, paths, calls = workspace
    (paths["backend"] / "requirements.txt").write_text("pytest\n")
    server = CapturingServer()
    developer_mcp.register_developer_tools(server)
    function = server.functions[tool_name]
    arguments = {"repository": "backend", "relative_path": None}
    if tool_name != "workspace_project_info":
        arguments["timeout_seconds"] = None
    result = function(**arguments).model_dump()

    assert result["ok"] is True
    assert result["repository"] == "backend"
    assert result.get("path", result.get("project_path")) == "."
    assert set(inspect.signature(function).parameters) == set(get_tool(tool_name)["parameters"])
    assert inspect.signature(function).parameters["repository"].default is None
    if calls:
        assert calls[0]["timeout_seconds"] == 60
        assert Path(calls[0]["cwd"]) == paths["backend"]


@pytest.mark.parametrize("tool_name", PROJECT_TOOLS)
def test_unrequested_repository_is_blocked_by_existing_grounding(workspace, tool_name):
    class NoExecution:
        async def call_tool(self, *args, **kwargs):
            pytest.fail("Ungrounded repository must not reach MCP")

    def no_approval(*args, **kwargs):
        pytest.fail("Ungrounded repository must not create an approval")

    gateway = ToolGateway(tool_lookup=get_tool, approval_creator=no_approval, mcp=NoExecution())
    agent = AgentDefinition(name="developer-specialist", description="Developer", model="test",
                            tools=list(PROJECT_TOOLS))
    result = asyncio.run(gateway.execute(
        agent=agent,
        user_input="Inspect or check the backend repository project.",
        tool_name=tool_name,
        arguments={"repository": "frontend", "relative_path": "project"},
    ))
    assert result["status"] == "denied"
    assert result["decision_code"] == "grounding_failed"


@pytest.mark.parametrize("tool_name", ("workspace_run_tests", "workspace_run_build"))
def test_repository_scoped_execution_still_requires_approval(workspace, tool_name):
    approvals = []

    def create_approval(tool_name, arguments, risk=None):
        approvals.append((tool_name, arguments, risk))
        return {"id": "test-approval"}

    class NoExecution:
        async def call_tool(self, *args, **kwargs):
            pytest.fail("Unapproved repository code must not reach MCP")

    gateway = ToolGateway(tool_lookup=get_tool, approval_creator=create_approval, mcp=NoExecution())
    agent = AgentDefinition(name="developer-specialist", description="Developer", model="test",
                            tools=list(PROJECT_TOOLS))
    arguments = {"repository": "backend", "relative_path": "project"}
    result = asyncio.run(gateway.execute(
        agent=agent, user_input="Run project checks in the backend repository.",
        tool_name=tool_name, arguments=arguments,
    ))
    assert result["status"] == "approval_required"
    assert approvals == [(tool_name, arguments, "medium")]
    assert "backend" in format_developer_execution_approval(arguments)


def test_repository_is_visible_in_presentations(workspace):
    info = developer.workspace_project_info(relative_path="project", repository="backend")
    execution = developer.workspace_run_tests(relative_path="project", repository="backend")
    assert "backend" in format_project_info_result(info)
    assert "backend" in format_developer_execution_result(execution)
    for card in (build_project_info_card(info), build_developer_execution_card(execution)):
        assert {"label": "Repository", "value": "backend"} in card["fields"]
