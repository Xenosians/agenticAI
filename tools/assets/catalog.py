from typing import (
    Any,
)


def format_asset_get_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    asset = (
        result.get(
            "asset"
        )
    )

    if not isinstance(
        asset,
        dict,
    ):

        return (
            "The asset lookup completed, but "
            "no readable asset record was returned."
        )

    asset_id = (
        asset.get(
            "asset_id",
            "Unknown asset",
        )
    )

    name = (
        asset.get(
            "name",
            "Unnamed asset",
        )
    )

    status = (
        asset.get(
            "status",
            "unknown",
        )
    )

    owner_id = (
        asset.get(
            "owner_id"
        )
    )

    owner_text = (
        owner_id
        if owner_id
        else "unassigned"
    )

    return (
        f"{asset_id} — {name}\n"
        f"- Status: {status}\n"
        f"- Owner: {owner_text}"
    )


def format_asset_search_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    assets = (
        result.get(
            "assets"
        )
    )

    if not isinstance(
        assets,
        list,
    ):

        return (
            "The asset search completed, but "
            "no readable result list was returned."
        )

    if not assets:

        return (
            "No assets matched the requested filters."
        )

    lines = []

    for asset in assets:

        if not isinstance(
            asset,
            dict,
        ):

            continue

        asset_id = (
            asset.get(
                "asset_id",
                "Unknown"
            )
        )

        name = (
            asset.get(
                "name",
                "Unnamed asset"
            )
        )

        status = (
            asset.get(
                "status",
                "unknown"
            )
        )

        owner = (
            asset.get(
                "owner_id"
            )
            or "unassigned"
        )

        lines.append(
            f"- {asset_id}: {name} "
            f"[{status}] owner={owner}"
        )

    return (
        "Matching assets:\n"
        + "\n".join(
            lines
        )
    )


def format_asset_mutation_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    message = (
        result.get(
            "message"
        )
    )

    if (
        isinstance(
            message,
            str,
        )
        and message.strip()
    ):

        return (
            message.strip()
        )

    return (
        "The asset assignment operation "
        "completed successfully."
    )


def format_asset_assign_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:

    asset_id = (
        arguments.get(
            "asset_id"
        )
    )

    user_id = (
        arguments.get(
            "user_id"
        )
    )

    if (
        isinstance(
            asset_id,
            str,
        )
        and asset_id.strip()
        and isinstance(
            user_id,
            str,
        )
        and user_id.strip()
    ):

        return (
            f"Assigning {asset_id} to {user_id} "
            "requires approval."
        )

    return (
        "Assigning the asset requires approval."
    )


def format_asset_unassign_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:

    asset_id = (
        arguments.get(
            "asset_id"
        )
    )

    if (
        isinstance(
            asset_id,
            str,
        )
        and asset_id.strip()
    ):

        return (
            f"Unassigning {asset_id} "
            "requires approval."
        )

    return (
        "Unassigning the asset requires approval."
    )


ASSET_TOOLS = {
    "asset_get": {
        "description": (
            "Retrieve the current inventory record for exactly "
            "one explicitly identified asset or device."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "asset_id",
        ],

        "parameters": {
            "asset_id": {
                "type":
                    "str",

                "description": (
                    "Exact asset identifier supplied by the user."
                ),
            },
        },

        "result_formatter":
            format_asset_get_result,
    },

    "asset_search": {
        "description": (
            "Search the managed asset inventory using bounded "
            "structured filters such as owner, asset type, status, "
            "or literal text."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "text",
            "owner_id",
            "asset_type",
            "status",
        ],

        "parameters": {
            "text": {
                "type":
                    "str",

                "description": (
                    "Optional exact text supplied by the user "
                    "for inventory search."
                ),
            },

            "owner_id": {
                "type":
                    "str",

                "description": (
                    "Optional exact owner identifier supplied "
                    "by the user."
                ),
            },

            "asset_type": {
                "type":
                    "str",

                "description": (
                    "Optional exact asset type supplied "
                    "by the user."
                ),
            },

            "status": {
                "type":
                    "str",

                "description": (
                    "Optional exact asset status supplied "
                    "by the user."
                ),
            },

            "limit": {
                "type":
                    "int",

                "description": (
                    "Maximum number of assets to return "
                    "from 1 to 25."
                ),
            },
        },

        "result_formatter":
            format_asset_search_result,
    },

    "asset_assign": {
        "description": (
            "Assign exactly one explicitly identified asset "
            "to exactly one explicitly identified user. "
            "This changes inventory ownership and requires approval."
        ),

        "risk":
            "medium",

        "requires_approval":
            True,

        "grounded_arguments": [
            "asset_id",
            "user_id",
        ],

        "parameters": {
            "asset_id": {
                "type":
                    "str",

                "description": (
                    "Exact asset identifier supplied by the user."
                ),
            },

            "user_id": {
                "type":
                    "str",

                "description": (
                    "Exact user identifier supplied by the user."
                ),
            },
        },

        "result_formatter":
            format_asset_mutation_result,

        "approval_formatter":
            format_asset_assign_approval,
    },

    "asset_unassign": {
        "description": (
            "Remove the current user assignment from exactly one "
            "explicitly identified asset. This changes inventory "
            "ownership and requires approval."
        ),

        "risk":
            "medium",

        "requires_approval":
            True,

        "grounded_arguments": [
            "asset_id",
        ],

        "parameters": {
            "asset_id": {
                "type":
                    "str",

                "description": (
                    "Exact asset identifier supplied by the user."
                ),
            },
        },

        "result_formatter":
            format_asset_mutation_result,

        "approval_formatter":
            format_asset_unassign_approval,
    },
}
