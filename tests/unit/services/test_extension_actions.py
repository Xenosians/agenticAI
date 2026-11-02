"""An extension command names a known action before any Lua worker or store sees it."""
import asyncio

import pytest

from services.extensions import service
from services.lua_adapters import runtime as adapter_runtime


def test_the_action_list_names_every_handled_action():
    assert {"list", "skill_save", "dataset", "dataset_crawl", "upload_finish", "workflow_validate"} <= service.ACTIONS
    assert all(isinstance(action, str) and action for action in service.ACTIONS)


def test_the_action_list_matches_the_actions_the_dispatcher_handles():
    import ast
    import inspect

    dispatcher = ast.parse(inspect.getsource(service._command))
    handled = {comparator.value for node in ast.walk(dispatcher)
               if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "action"
               for operator, comparator in zip(node.ops, node.comparators)
               if isinstance(operator, ast.Eq) and isinstance(comparator, ast.Constant)}
    assert handled == service.ACTIONS


@pytest.mark.parametrize("action", ["bogus", "skill_", "upload_anything", "dataset_secret", ""])
def test_an_unknown_action_is_refused_before_any_lua_worker_starts(monkeypatch, action):
    def no_worker(*args, **kwargs):
        raise AssertionError("a Lua worker was used for an unknown action")

    monkeypatch.setattr(adapter_runtime.WORKERS, "exchange", no_worker)
    monkeypatch.setattr(service, "extension_store", no_worker)
    with pytest.raises(ValueError, match="Unknown extension action"):
        asyncio.run(service.command("owner-1", action, {}, settings=None))
