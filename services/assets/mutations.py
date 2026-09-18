from abc import (
    ABC,
    abstractmethod,
)

from .base import (
    AssetService,
)

from .mock import (
    MockAssetService,
)

from .types import (
    AssetMutationResult,
)


class AssetMutationService(
    ABC
):

    @abstractmethod
    def assign_asset(
        self,
        asset_id: str,
        user_id: str,
    ) -> AssetMutationResult:
        raise NotImplementedError

    @abstractmethod
    def unassign_asset(
        self,
        asset_id: str,
    ) -> AssetMutationResult:
        raise NotImplementedError


class MockAssetMutationService(
    AssetMutationService
):

    def __init__(
        self,
        assets: MockAssetService,
    ) -> None:

        self.assets = (
            assets
        )

    @staticmethod
    def _normalize_user_id(
        user_id: str,
    ) -> tuple[
        str | None,
        str | None,
    ]:

        if not isinstance(
            user_id,
            str,
        ):

            return (
                None,
                "user_id must be a string.",
            )

        normalized = (
            user_id
            .strip()
            .lower()
        )

        if not normalized:

            return (
                None,
                "user_id must not be empty.",
            )

        return (
            normalized,
            None,
        )

    def assign_asset(
        self,
        asset_id: str,
        user_id: str,
    ) -> AssetMutationResult:

        lookup = (
            self.assets
            .get_asset(
                asset_id
            )
        )

        if (
            not lookup.ok
            or lookup.asset
            is None
        ):

            return (
                AssetMutationResult(
                    ok=False,

                    status=(
                        lookup.status
                    ),

                    provider="mock",

                    error=(
                        lookup.error
                    ),
                )
            )

        (
            normalized_user,
            user_error,
        ) = (
            self._normalize_user_id(
                user_id
            )
        )

        if normalized_user is None:

            return (
                AssetMutationResult(
                    ok=False,

                    status="denied",

                    provider="mock",

                    asset_id=(
                        lookup.asset.asset_id
                    ),

                    error=(
                        user_error
                    ),
                )
            )

        asset = (
            lookup.asset
        )

        if (
            asset.status
            in {
                "repair",
                "retired",
            }
        ):

            return (
                AssetMutationResult(
                    ok=False,

                    status="denied",

                    provider="mock",

                    asset_id=(
                        asset.asset_id
                    ),

                    owner_id=(
                        asset.owner_id
                    ),

                    error=(
                        "The asset cannot be assigned "
                        f"while its status is '{asset.status}'."
                    ),
                )
            )

        if (
            asset.owner_id
            == normalized_user
        ):

            return (
                AssetMutationResult(
                    ok=True,

                    status="executed",

                    provider="mock",

                    changed=False,

                    asset_id=(
                        asset.asset_id
                    ),

                    owner_id=(
                        normalized_user
                    ),

                    message=(
                        f"Asset '{asset.asset_id}' "
                        "was already assigned to "
                        f"'{normalized_user}'."
                    ),
                )
            )

        asset.owner_id = (
            normalized_user
        )

        if (
            asset.status
            == "available"
        ):

            asset.status = (
                "in_use"
            )

        return (
            AssetMutationResult(
                ok=True,

                status="executed",

                provider="mock",

                changed=True,

                asset_id=(
                    asset.asset_id
                ),

                owner_id=(
                    normalized_user
                ),

                message=(
                    f"Asset '{asset.asset_id}' "
                    "was assigned to "
                    f"'{normalized_user}'."
                ),
            )
        )

    def unassign_asset(
        self,
        asset_id: str,
    ) -> AssetMutationResult:

        lookup = (
            self.assets
            .get_asset(
                asset_id
            )
        )

        if (
            not lookup.ok
            or lookup.asset
            is None
        ):

            return (
                AssetMutationResult(
                    ok=False,

                    status=(
                        lookup.status
                    ),

                    provider="mock",

                    error=(
                        lookup.error
                    ),
                )
            )

        asset = (
            lookup.asset
        )

        if (
            asset.owner_id
            is None
        ):

            return (
                AssetMutationResult(
                    ok=True,

                    status="executed",

                    provider="mock",

                    changed=False,

                    asset_id=(
                        asset.asset_id
                    ),

                    owner_id=None,

                    message=(
                        f"Asset '{asset.asset_id}' "
                        "was already unassigned."
                    ),
                )
            )

        previous_owner = (
            asset.owner_id
        )

        asset.owner_id = (
            None
        )

        if (
            asset.status
            == "in_use"
        ):

            asset.status = (
                "available"
            )

        return (
            AssetMutationResult(
                ok=True,

                status="executed",

                provider="mock",

                changed=True,

                asset_id=(
                    asset.asset_id
                ),

                owner_id=None,

                message=(
                    f"Asset '{asset.asset_id}' "
                    "was unassigned from "
                    f"'{previous_owner}'."
                ),
            )
        )


def build_asset_mutation_service(
    assets: AssetService,
) -> AssetMutationService:

    if isinstance(
        assets,
        MockAssetService,
    ):

        return (
            MockAssetMutationService(
                assets
            )
        )

    raise RuntimeError(
        "Unsupported AssetService implementation "
        "for asset mutations: "
        f"{type(assets).__name__}"
    )
