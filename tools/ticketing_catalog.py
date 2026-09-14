from tools.ticketing_presentation import (
    build_ticket_get_card,
    format_ticket_get_result,
)


TICKETING_TOOLS = {
    "ticket_get": {
        "description": (
            "Retrieve one ticket by its exact "
            "ticket identifier from the configured "
            "ticketing provider."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "ticket_key",
        ],

        "parameters": {
            "ticket_key": {
                "type":
                    "str",

                "description": (
                    "Exact ticket identifier supplied "
                    "by the user, for example ITSM-101."
                ),
            },
        },

        "result_formatter": (
            format_ticket_get_result
        ),

        "presentation_builder": (
            build_ticket_get_card
        ),
    },
}