"""Owner switches for MCP tools.

An owner can turn a registered tool off for their own account, one tool at a
time or a whole package (the tools of one provider, as grouped by
``services.integration_readiness``). A turned-off tool is refused by
ToolGateway for that owner's requests before policy, approval or provider
(decision ``tool_disabled_by_owner``); other owners are unaffected. The switch
is the owner's preference, not an authority: turning a tool on grants nothing
the tool's policy, approval and provider checks do not already allow.
"""
from __future__ import annotations

from datetime import datetime, timezone

from services.extensions.store import ExtensionStore


class ToolSwitchError(ValueError):
    """A switch request was refused: a stable code and a fixed message."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def public_detail(self) -> dict:
        return {"code": self.code, "message": self.message}


def _registered(name) -> bool:
    from tools.registry import TOOLS

    return isinstance(name, str) and name in TOOLS


def _package_tools(package) -> list[str] | None:
    """The registered tools of one package, or None for an unknown package."""
    from services.integration_readiness import PROVIDER_FAMILIES
    from tools.registry import TOOLS

    for family in PROVIDER_FAMILIES:
        if isinstance(package, str) and family.id == package:
            return sorted(name for name in family.capabilities if name in TOOLS)
    return None


class ToolSwitches:
    """Each owner's turned-off tools, in the extension store's database (absent row = on)."""

    def __init__(self, store: ExtensionStore):
        self.store = store
        with store.connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS tool_switch (
                    owner TEXT NOT NULL, tool TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner, tool))
            """)

    def set_enabled(self, owner: str, name, enabled) -> dict:
        if not _registered(name):
            raise ToolSwitchError("tool_not_found", "Tool not found.")
        if not isinstance(enabled, bool):
            raise ToolSwitchError("tool_enabled_invalid", "enabled must be true or false.")
        with self.store.connect() as db:
            if enabled:
                db.execute("DELETE FROM tool_switch WHERE owner=? AND tool=?", (owner, name))
            else:
                db.execute("INSERT OR REPLACE INTO tool_switch(owner, tool, updated_at) VALUES (?,?,?)",
                           (owner, name, datetime.now(timezone.utc).isoformat()))
        return {"id": name, "enabled": enabled}

    def set_package_enabled(self, owner: str, package, enabled) -> dict:
        """Turn every tool of one package on or off in one write."""
        tools = _package_tools(package)
        if tools is None:
            raise ToolSwitchError("tool_package_not_found", "Tool package not found.")
        if not isinstance(enabled, bool):
            raise ToolSwitchError("tool_enabled_invalid", "enabled must be true or false.")
        now = datetime.now(timezone.utc).isoformat()
        with self.store.connect() as db:
            if enabled:
                db.executemany("DELETE FROM tool_switch WHERE owner=? AND tool=?",
                               [(owner, name) for name in tools])
            else:
                db.executemany("INSERT OR REPLACE INTO tool_switch(owner, tool, updated_at) VALUES (?,?,?)",
                               [(owner, name, now) for name in tools])
        return {"package": package, "enabled": enabled, "tools": tools}

    def disabled(self, owner: str) -> frozenset[str]:
        with self.store.connect() as db:
            rows = db.execute("SELECT tool FROM tool_switch WHERE owner=?", (owner,)).fetchall()
        return frozenset(row["tool"] for row in rows)
