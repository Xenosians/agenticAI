from pathlib import Path
from types import SimpleNamespace

from services.lua_adapters import plugins
from services.lua_adapters.runtime import LuaAdapterError


def test_plugin_catalog_reports_resolved_sources_without_host_paths(monkeypatch):
    def resolve(plugin_id):
        if plugin_id == "tool.default":
            return SimpleNamespace(packaged=False, sha256="a" * 64, path=Path("/private/profile/tool.lua"))
        return SimpleNamespace(packaged=True, sha256="b" * 64, path=Path("/opt/itsm/plugin.lua"))

    monkeypatch.setattr(plugins, "resolve_plugin", resolve)
    catalog = plugins.plugin_catalog()

    assert catalog["schema"] == "lua-plugin-catalog.v1"
    assert catalog["scope"] == "runtime"
    assert len(catalog["plugins"]) == len(plugins.PLUGIN_FILES)
    packaged = next(row for row in catalog["plugins"] if row["id"] == "model.local")
    selected = next(row for row in catalog["plugins"] if row["id"] == "tool.default")
    assert packaged == {"id": "model.local", "protocol": "model", "state": "resolved",
                        "source": "packaged", "sha256": "b" * 64}
    assert selected["source"] == "machine_profile"
    assert selected["state"] == "resolved"
    assert "/private" not in repr(catalog)
    assert "/opt/" not in repr(catalog)


def test_plugin_catalog_preserves_invalid_profile_as_unavailable_without_fallback(monkeypatch):
    def resolve(plugin_id):
        if plugin_id == "model.local":
            raise LuaAdapterError("lua_adapter_configuration_invalid")
        return SimpleNamespace(packaged=True, sha256="c" * 64, path=Path("/opt/itsm/plugin.lua"))

    monkeypatch.setattr(plugins, "resolve_plugin", resolve)
    row = next(row for row in plugins.plugin_catalog()["plugins"] if row["id"] == "model.local")

    assert row == {"id": "model.local", "protocol": "model", "state": "unavailable",
                   "source": "unavailable", "reason_code": "lua_adapter_configuration_invalid"}
