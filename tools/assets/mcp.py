from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)

from mcp.server import (
    MCPServer,
)

from services.assets import (
    AssetMutationService,
    AssetSearchQuery,
    AssetService,
)


class AssetRecordMCPResult(
    BaseModel
):
    provider: str

    asset_id: str
    name: str
    asset_type: str
    status: str

    owner_id: (
        str | None
    ) = None

    serial_number: (
        str | None
    ) = None

    platform: (
        str | None
    ) = None

    location: (
        str | None
    ) = None


class AssetLookupMCPResult(
    BaseModel
):
    ok: bool
    status: str

    asset: (
        AssetRecordMCPResult
        | None
    ) = None

    error: (
        str | None
    ) = None


class AssetSearchMCPResult(
    BaseModel
):
    ok: bool
    status: str

    assets: list[
        AssetRecordMCPResult
    ] = Field(
        default_factory=list
    )

    count: int = 0
    truncated: bool = False

    error: (
        str | None
    ) = None


class AssetMutationMCPResult(
    BaseModel
):
    ok: bool
    status: str

    provider: (
        str | None
    ) = None

    changed: bool = False

    asset_id: (
        str | None
    ) = None

    owner_id: (
        str | None
    ) = None

    message: (
        str | None
    ) = None

    error: (
        str | None
    ) = None


def register_asset_tools(
    server: MCPServer,
    assets: AssetService,
    mutations: AssetMutationService,
) -> None:

    @server.tool()
    def asset_get(
        asset_id: str,
    ) -> AssetLookupMCPResult:

        result = (
            assets
            .get_asset(
                asset_id
            )
        )

        return (
            AssetLookupMCPResult(
                **result.model_dump()
            )
        )

    @server.tool()
    def asset_search(
        text: str | None = None,
        owner_id: str | None = None,
        asset_type: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> AssetSearchMCPResult:

        try:

            query = (
                AssetSearchQuery(
                    text=text,

                    owner_id=(
                        owner_id
                    ),

                    asset_type=(
                        asset_type
                    ),

                    status=status,

                    limit=(
                        10
                        if limit is None
                        else limit
                    ),
                )
            )

        except ValidationError:

            return (
                AssetSearchMCPResult(
                    ok=False,

                    status="denied",

                    error=(
                        "Invalid asset search filters."
                    ),
                )
            )

        result = (
            assets
            .search_assets(
                query
            )
        )

        return (
            AssetSearchMCPResult(
                **result.model_dump()
            )
        )

    @server.tool()
    def asset_assign(
        asset_id: str,
        user_id: str,
    ) -> AssetMutationMCPResult:

        result = (
            mutations
            .assign_asset(
                asset_id,
                user_id,
            )
        )

        return (
            AssetMutationMCPResult(
                **result.model_dump()
            )
        )

    @server.tool()
    def asset_unassign(
        asset_id: str,
    ) -> AssetMutationMCPResult:

        result = (
            mutations
            .unassign_asset(
                asset_id
            )
        )

        return (
            AssetMutationMCPResult(
                **result.model_dump()
            )
        )
