import os
import subprocess
from pathlib import Path

import pytest

from services.extensions.harness import _resolve_node_runtime


@pytest.fixture(autouse=True)
def isolated_node_configuration(monkeypatch):
    monkeypatch.delenv("PI_NODE_BINARY", raising=False)

    def unexpected_process(*args, **kwargs):
        pytest.fail("Node resolver tests must not execute a host process")

    monkeypatch.setattr(subprocess, "run", unexpected_process)


@pytest.fixture
def executable_node(tmp_path):
    binary = tmp_path / "installed-node" / "bin" / "node"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary


def test_configured_executable_bypasses_host_node_lookup(monkeypatch, executable_node):
    monkeypatch.setenv("PI_NODE_BINARY", f"  {executable_node}  ")

    assert _resolve_node_runtime() == executable_node.resolve()


def test_configured_home_relative_executable_expands_before_isolation(
    monkeypatch, executable_node
):
    monkeypatch.setenv("HOME", str(executable_node.parent))
    monkeypatch.setenv("USERPROFILE", str(executable_node.parent))
    monkeypatch.setenv("PI_NODE_BINARY", "~/node")

    assert _resolve_node_runtime() == executable_node.resolve()


@pytest.mark.parametrize("invalid_kind", ["relative", "missing", "directory", "nonexecutable"])
def test_configured_runtime_rejects_unusable_paths(
    monkeypatch, tmp_path, executable_node, invalid_kind
):
    candidates = {
        "relative": Path("node"),
        "missing": tmp_path / "missing-node",
        "directory": executable_node.parent,
        "nonexecutable": executable_node,
    }
    if invalid_kind == "nonexecutable":
        monkeypatch.setattr(os, "access", lambda path, mode: False)
    monkeypatch.setenv("PI_NODE_BINARY", str(candidates[invalid_kind]))

    with pytest.raises(ValueError, match="PI_NODE_BINARY must point to an executable Node binary"):
        _resolve_node_runtime()


def test_host_shim_resolves_real_executable_with_original_home(
    monkeypatch, tmp_path, executable_node
):
    original_home = str(tmp_path / "host-home")
    monkeypatch.setenv("HOME", original_home)

    def host_node(command, **options):
        assert command == ["node", "-p", "process.execPath"]
        assert options["env"]["HOME"] == original_home
        assert options["stdin"] == subprocess.DEVNULL
        assert options["timeout"] == 10
        return subprocess.CompletedProcess(command, 0, f"{executable_node}\n", "")

    monkeypatch.setattr(subprocess, "run", host_node)

    assert _resolve_node_runtime() == executable_node.resolve()


@pytest.mark.parametrize("failure", ["missing_node", "timeout", "shim_failure"])
def test_host_node_failure_has_safe_actionable_error(monkeypatch, failure):
    diagnostic = "private-host-diagnostic"

    def broken_node(command, **options):
        if failure == "missing_node":
            raise FileNotFoundError(diagnostic)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, options["timeout"], stderr=diagnostic)
        return subprocess.CompletedProcess(command, 1, "", diagnostic)

    monkeypatch.setattr(subprocess, "run", broken_node)

    with pytest.raises(ValueError, match="Pi requires a working Node runtime on the AI host") as exc:
        _resolve_node_runtime()
    assert diagnostic not in str(exc.value)


def test_host_node_without_executable_path_is_rejected(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **options: subprocess.CompletedProcess(command, 0, " \n", ""),
    )

    with pytest.raises(ValueError, match="Unable to resolve Pi Node runtime"):
        _resolve_node_runtime()


@pytest.mark.parametrize("invalid_kind", ["relative", "missing", "directory", "nonexecutable"])
def test_host_node_rejects_unusable_reported_paths(
    monkeypatch, tmp_path, executable_node, invalid_kind
):
    candidates = {
        "relative": Path("node"),
        "missing": tmp_path / "missing-node",
        "directory": executable_node.parent,
        "nonexecutable": executable_node,
    }
    if invalid_kind == "nonexecutable":
        monkeypatch.setattr(os, "access", lambda path, mode: False)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **options: subprocess.CompletedProcess(
            command, 0, str(candidates[invalid_kind]), ""
        ),
    )

    with pytest.raises(ValueError, match="Resolved Pi Node runtime is invalid"):
        _resolve_node_runtime()
