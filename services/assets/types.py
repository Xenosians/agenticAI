from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class AssetRecord(
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


class AssetLookupResult(
    BaseModel
):
    ok: bool
    status: str

    asset: (
        AssetRecord | None
    ) = None

    error: (
        str | None
    ) = None


class AssetSearchQuery(
    BaseModel
):
    text: (
        str | None
    ) = Field(
        default=None,
        max_length=200,
    )

    owner_id: (
        str | None
    ) = Field(
        default=None,
        max_length=100,
    )

    asset_type: (
        str | None
    ) = Field(
        default=None,
        max_length=100,
    )

    status: (
        str | None
    ) = Field(
        default=None,
        max_length=100,
    )

    limit: int = Field(
        default=10,
        ge=1,
        le=25,
    )

    @field_validator(
        "text",
        "owner_id",
        "asset_type",
        "status",
        mode="before",
    )
    @classmethod
    def normalize_optional_string(
        cls,
        value,
    ):

        if value is None:

            return None

        if not isinstance(
            value,
            str,
        ):

            raise ValueError(
                "Asset search filters "
                "must be strings."
            )

        normalized = (
            value.strip()
        )

        if not normalized:

            return None

        return normalized


class AssetSearchResult(
    BaseModel
):
    ok: bool
    status: str

    assets: list[
        AssetRecord
    ] = Field(
        default_factory=list
    )

    count: int = Field(
        default=0,
        ge=0,
    )

    truncated: bool = False

    error: (
        str | None
    ) = None


class AssetMutationResult(
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
