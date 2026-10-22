"""Schema imports and introspection cannot implicitly start a deployment."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[3]


def child(code):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"}
    completed = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=ROOT,
                               env=env, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr


def test_imports_do_not_load_dotenv_or_construct_settings():
    child('''
        from unittest.mock import patch
        with patch("dotenv.dotenv_values", side_effect=AssertionError("implicit dotenv read")):
            from config import Settings
            with patch.object(Settings, "__init__", side_effect=AssertionError("implicit startup")):
                import mcp_server
                import api.app
                assert not hasattr(mcp_server, "mcp")
    ''')


def test_signature_inventory_ignores_deployment_and_never_dispatches():
    child('''
        import os
        from unittest.mock import patch
        os.environ["DIRECTORY_BACKEND"] = "ldap"
        os.environ["TICKETING_BACKEND"] = "jira"
        os.environ["ASSET_BACKEND"] = "sqlite"
        os.environ["KNOWLEDGE_BACKEND"] = "local"
        with patch("dotenv.dotenv_values", side_effect=AssertionError("implicit dotenv read")):
            from mcp_server import register_capability_signatures
            from tools.runtime_schema import clear_runtime_schemas, runtime_registered_tools
            from tools.registry import TOOLS
            with patch("services.directory.mock.MockDirectoryService.account_status", side_effect=AssertionError("tool dispatch")), \\
                 patch("httpx.Client.send", side_effect=AssertionError("provider request")), \\
                 patch("socket.socket.connect", side_effect=AssertionError("network access")):
                clear_runtime_schemas()
                register_capability_signatures()
                assert set(runtime_registered_tools()) == set(TOOLS)
    ''')


def test_explicit_startup_still_exports_policy_before_constructing_server(monkeypatch):
    import mcp_server
    import config.settings
    calls = []
    class Server:
        def run(self):
            calls.append("run")
    monkeypatch.setattr(config.settings, "export_process_env_from_dotenv", lambda: calls.append("policy"))
    def construct():
        calls.append("construct")
        return Server()
    monkeypatch.setattr(mcp_server, "create_mcp_server", construct)
    mcp_server.main()
    assert calls == ["policy", "construct", "run"]
