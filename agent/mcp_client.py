import sys

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
    ) -> dict:
        if (
            self._client
            is None
        ):
            raise RuntimeError(
                "MCP runtime has not "
                "been started."
            )

        try:
            result = (
                await
                self._client
                .call_tool(
                    tool_name,
                    arguments,
                )
            )

        except Exception as exc:
            return {
                "ok":
                    False,

                "status":
                    "error",

                "error": (
                    "MCP call failed: "
                    f"{exc}"
                ),
            }

        if result.is_error:
            return {
                "ok":
                    False,

                "status":
                    "error",

                "error": (
                    f"MCP tool '{tool_name}' "
                    "returned an error."
                ),
            }

        if (
            result
            .structured_content
            is None
        ):
            return {
                "ok":
                    False,

                "status":
                    "error",

                "error": (
                    f"MCP tool '{tool_name}' "
                    "returned no structured "
                    "content."
                ),
            }

        return (
            result
            .structured_content
        )