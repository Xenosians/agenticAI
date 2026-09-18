from abc import (
    ABC,
    abstractmethod,
)

from .types import (
    AssetLookupResult,
    AssetSearchQuery,
    AssetSearchResult,
)


class AssetService(
    ABC
):

    @abstractmethod
    def get_asset(
        self,
        asset_id: str,
    ) -> AssetLookupResult:
        raise NotImplementedError

    @abstractmethod
    def search_assets(
        self,
        query: AssetSearchQuery,
    ) -> AssetSearchResult:
        raise NotImplementedError
