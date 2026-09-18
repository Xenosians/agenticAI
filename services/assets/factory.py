from .base import (
    AssetService,
)

from .mock import (
    MockAssetService,
)


def build_asset_service(
) -> AssetService:
    """
    Build the current asset inventory provider.

    The provider-neutral boundary is established now.

    External CMDB / MDM providers can be added behind this factory
    without changing model-facing capabilities.
    """

    return (
        MockAssetService()
    )
