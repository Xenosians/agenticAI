from pathlib import Path
import json

from learning.training.specialist_sft import load_training_config
from subagents.core.definitions.loader import load_agent_definition

CONFIG = Path("learning/config/specialists/developer-shell-v3.json")


def test_shell_v3_continues_from_promoted_developer_profile():
    config = load_training_config(CONFIG)
    assert config.base_model_key == "developer-func-trained"
    assert config.candidate_model_key == "developer-func-trained-shell-v3"


def test_shell_v3_tools_are_declared_by_developer_agent():
    config = load_training_config(CONFIG)
    agent = load_agent_definition(Path(config.agent_definition))
    allowed = set(agent.tools)
    configured = {x.tool_name for x in config.examples} | {
        x.expected_tool for x in config.eval_cases
    }
    assert configured <= allowed


def test_shell_v3_covers_dedicated_workspace_semantics():
    config = load_training_config(CONFIG)
    tools = {x.tool_name for x in config.examples}
    required = {
        "workspace_read_text",
        "workspace_list",
        "workspace_search",
        "workspace_file_info",
        "workspace_project_info",
        "workspace_run_tests",
        "workspace_run_build",
        "workspace_process_snapshot",
        "workspace_service_status",
        "workspace_service_logs",
        "workspace_mkdir",
        "process_exec",
        "workspace_git_status",
        "workspace_git_commit",
        "workspace_git_push",
    }
    assert required <= tools


def test_process_exec_training_stays_inside_known_v2_native_shapes():
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    items = [x for x in payload["examples"] if x["tool_name"] == "process_exec"]
    assert items
    allowed = {"pwd", "ls", "mkdir", "git"}
    denied_tokens = {"rm", "sudo", "curl", "wget", "chmod"}
    for item in items:
        args = item["arguments"]
        executable = args["executable"]
        argv = args["args"]
        assert executable in allowed
        assert not denied_tokens.intersection({executable, *argv})
        if executable == "git":
            assert argv == ["branch", "--list", "--no-color"]


def test_shell_eval_has_semantic_vs_exact_command_contrasts():
    config = load_training_config(CONFIG)
    names = {x.name for x in config.eval_cases}
    assert {
        "tests dedicated not process exec",
        "build dedicated not process exec",
        "semantic mkdir dedicated",
        "explicit native pwd",
        "semantic git status not native git",
        "git commit not push",
        "git push not commit",
    } <= names
