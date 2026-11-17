import sys
import os

from pathlib import Path

from mcp import (
    Client,
    StdioServerParameters,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DEFAULT_MCP_SERVER_PATH = (
    PROJECT_ROOT
    / "mcp_server.py"
)

# The Janus account a tool call serves, sent as request metadata (never as a
# model argument). Each account connects its own Proxmox and Tailscale, so the
# server's tools read the connection of exactly this owner.
OWNER_META_KEY = "janus.itsm/owner"


def _unconfirmed_result(code: str) -> dict:
    """A protocol failure after submission says nothing about a tool's effect.

    The server may have acted before raising, losing its reply or failing output
    validation. ApprovalManager must keep its durable execution claim unresolved.
    Raw exception/SDK text can contain arguments or provider credentials.
    """
    return {
        "ok": False,
        "status": "outcome_unknown",
        "error_code": code,
        "error": (
            "The MCP tool outcome could not be confirmed. "
            "Reconcile the operation before attempting another execution."
        ),
    }


class MCPRuntime:
    """
    Explicit lifecycle owner for the persistent MCP subprocess.

    MCPRuntime is an ordinary runtime object.

    FastAPI lifespan owns one instance for the AI service.
    Tests may create their own isolated instances.

    There is intentionally no module-global MCP runtime.
    """

    def __init__(
        self,
        server_path: (
            str
            | Path
            | None
        ) = None,
    ) -> None:
        if server_path is None:
            path = (
                DEFAULT_MCP_SERVER_PATH
            )

        else:
            path = (
                Path(
                    server_path
                )
                .expanduser()
            )

            if not path.is_absolute():
                path = (
                    PROJECT_ROOT
                    / path
                )

        self.server_path = (
            path.resolve()
        )

        self._context = None

        self._client = None

    @property
    def started(
        self,
    ) -> bool:
        return (
            self._client
            is not None
        )

    async def start(
        self,
    ) -> None:
        if (
            self._client
            is not None
        ):
            return

        if not (
            self.server_path
            .exists()
        ):
            raise RuntimeError(
                "MCP server does not exist: "
                f"{self.server_path}"
            )

        server = (
            StdioServerParameters(
                command=(
                    sys.executable
                ),

                args=[
                    str(
                        self.server_path
                    )
                ],
                # The trusted provider server needs the reviewed application
                # settings in env-only deployments. SDK defaults inherit only
                # shell basics. Isolated Lua workers keep their separate,
                # credential-free environment.
                env=dict(os.environ),
                cwd=str(PROJECT_ROOT),
            )
        )

        self._context = (
            Client(
                server
            )
        )

        self._client = (
            await
            self._context
            .__aenter__()
        )

        print(
            "[MCP] Persistent server started."
        )

    async def stop(
        self,
    ) -> None:
        if (
            self._context
            is None
        ):
            return

        try:
            await (
                self._context
                .__aexit__(
                    None,
                    None,
                    None,
                )
            )

        finally:
            self._context = (
                None
            )

            self._client = (
                None
            )

        print(
            "[MCP] Server stopped."
        )

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
        *,
        owner: str | None = None,
    ) -> dict:
        """Call one tool for ``owner``, or for the owner of the executing request.

        An approved action passes the owner stored with its approval, so it runs
        with the connection of the account that proposed it.
        """
        if (
            self._client
            is None
        ):
            raise RuntimeError(
                "MCP runtime has not "
                "been started."
            )

        if owner is None:
            from services.extensions.context import current_owner
            owner = current_owner()
        # No owner, no metadata: such a call is sent exactly as before.
        owner_meta = {"meta": {OWNER_META_KEY: owner}} if owner else {}

        try:
            result = (
                await
                self._client
                .call_tool(
                    tool_name,
                    arguments,
                    **owner_meta,
                )
            )

        except Exception:
            return _unconfirmed_result("mcp_call_unconfirmed")

        if result.is_error:
            return _unconfirmed_result("mcp_tool_error")

        if (
            result
            .structured_content
            is None
        ):
            return _unconfirmed_result("mcp_structured_receipt_missing")

        if not isinstance(result.structured_content, dict):
            return _unconfirmed_result("mcp_structured_receipt_invalid")

        return (
            result
            .structured_content
        )
