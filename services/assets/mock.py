from .base import (
    AssetService,
)

from .types import (
    AssetLookupResult,
    AssetRecord,
    AssetSearchQuery,
    AssetSearchResult,
)


class MockAssetService(
    AssetService
):

    def __init__(
        self,
    ) -> None:

        self.assets = {
            "LAP-001": (
                AssetRecord(
                    provider="mock",

                    asset_id="LAP-001",

                    name=(
                        "Engineering Laptop"
                    ),

                    asset_type="laptop",

                    status="in_use",

                    owner_id="jdoe",

                    serial_number=(
                        "MOCK-LAP-001"
                    ),

                    platform=(
                        "Windows 11"
                    ),

                    location=(
                        "Jakarta HQ"
                    ),
                )
            ),

            "LAP-002": (
                AssetRecord(
                    provider="mock",

                    asset_id="LAP-002",

                    name=(
                        "Service Desk Spare Laptop"
                    ),

                    asset_type="laptop",

                    status="available",

                    owner_id=None,

                    serial_number=(
                        "MOCK-LAP-002"
                    ),

                    platform=(
                        "Windows 11"
                    ),

                    location=(
                        "Jakarta HQ"
                    ),
                )
            ),

            "PHONE-001": (
                AssetRecord(
                    provider="mock",

                    asset_id="PHONE-001",

                    name=(
                        "Corporate Mobile Phone"
                    ),

                    asset_type="phone",

                    status="in_use",

                    owner_id="asmith",

                    serial_number=(
                        "MOCK-PHONE-001"
                    ),

                    platform="Android",

                    location=(
                        "Jakarta HQ"
                    ),
                )
            ),

            "LAP-003": (
                AssetRecord(
                    provider="mock",

                    asset_id="LAP-003",

                    name=(
                        "Repair Queue Laptop"
                    ),

                    asset_type="laptop",

                    status="repair",

                    owner_id=None,

                    serial_number=(
                        "MOCK-LAP-003"
                    ),

                    platform=(
                        "Windows 11"
                    ),

                    location=(
                        "Service Desk"
                    ),
                )
            ),
        }

    @staticmethod
    def _normalize_asset_id(
        asset_id: str,
    ) -> tuple[
        str | None,
        str | None,
    ]:

        if not isinstance(
            asset_id,
            str,
        ):

            return (
                None,
                "asset_id must be a string.",
            )

        normalized = (
            asset_id
            .strip()
            .upper()
        )

        if not normalized:

            return (
                None,
                "asset_id must not be empty.",
            )

        return (
            normalized,
            None,
        )

    def get_asset(
        self,
        asset_id: str,
    ) -> AssetLookupResult:

        (
            normalized_id,
            error,
        ) = (
            self._normalize_asset_id(
                asset_id
            )
        )

        if normalized_id is None:

            return (
                AssetLookupResult(
                    ok=False,

                    status="denied",

                    error=error,
                )
            )

        asset = (
            self.assets.get(
                normalized_id
            )
        )

        if asset is None:

            return (
                AssetLookupResult(
                    ok=False,

                    status="not_found",

                    error=(
                        f"Asset '{normalized_id}' "
                        "was not found."
                    ),
                )
            )

        return (
            AssetLookupResult(
                ok=True,

                status="success",

                asset=asset,

                error=None,
            )
        )

    def search_assets(
        self,
        query: AssetSearchQuery,
    ) -> AssetSearchResult:

        if not isinstance(
            query,
            AssetSearchQuery,
        ):

            return (
                AssetSearchResult(
                    ok=False,

                    status="denied",

                    error=(
                        "Invalid asset search query."
                    ),
                )
            )

        assets = list(
            self.assets.values()
        )

        if query.text is not None:

            needle = (
                query.text.casefold()
            )

            assets = [
                asset

                for asset
                in assets

                if any(
                    needle
                    in value.casefold()

                    for value
                    in [
                        asset.asset_id,
                        asset.name,
                        (
                            asset.serial_number
                            or ""
                        ),
                        (
                            asset.platform
                            or ""
                        ),
                        (
                            asset.location
                            or ""
                        ),
                    ]
                )
            ]

        if query.owner_id is not None:

            owner_id = (
                query
                .owner_id
                .casefold()
            )

            assets = [
                asset

                for asset
                in assets

                if (
                    asset.owner_id
                    is not None
                    and asset
                    .owner_id
                    .casefold()
                    == owner_id
                )
            ]

        if query.asset_type is not None:

            asset_type = (
                query
                .asset_type
                .casefold()
            )

            assets = [
                asset

                for asset
                in assets

                if (
                    asset
                    .asset_type
                    .casefold()
                    == asset_type
                )
            ]

        if query.status is not None:

            status = (
                query
                .status
                .casefold()
            )

            assets = [
                asset

                for asset
                in assets

                if (
                    asset
                    .status
                    .casefold()
                    == status
                )
            ]

        assets.sort(
            key=lambda asset: (
                asset.asset_id
            )
        )

        truncated = (
            len(
                assets
            )
            > query.limit
        )

        selected = (
            assets[
                :query.limit
            ]
        )

        return (
            AssetSearchResult(
                ok=True,

                status="success",

                assets=selected,

                count=len(
                    selected
                ),

                truncated=truncated,

                error=None,
            )
        )
