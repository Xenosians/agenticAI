from tools.developer.presentation import (
    build_developer_execution_card,
    build_project_info_card,
    format_developer_execution_approval,
    format_developer_execution_result,
    format_project_info_result,
)

from tools.developer.runtime_presentation import (
    build_process_snapshot_card,
    build_service_logs_card,
    build_service_status_card,
    format_process_snapshot_result,
    format_service_logs_result,
    format_service_status_result,
)


DEVELOPER_TOOLS = {
    # ============================================================
    # PROJECT DETECTION / EXECUTION
    # ============================================================

    "workspace_project_info": {
        "description": (
            "Detect the supported project type and "
            "available governed developer operations "
            "at a workspace directory."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments":
            [],

        "parameters": {
            "relative_path": {
                "type":
                    "str",

                "description": (
                    "Optional workspace-relative project "
                    "directory. Defaults to the workspace root."
                ),
            },
        },

        "result_formatter": (
            format_project_info_result
        ),

        "presentation_builder": (
            build_project_info_card
        ),
    },

    "workspace_run_tests": {
        "description": (
            "Run the trusted test command for a detected "
            "Python, Elixir, or Nim project."
        ),

        "risk":
            "medium",

        "requires_approval":
            True,

        "grounded_arguments":
            [],

        "parameters": {
            "relative_path": {
                "type":
                    "str",

                "description": (
                    "Optional workspace-relative project "
                    "directory. Defaults to the workspace root."
                ),
            },

            "timeout_seconds": {
                "type":
                    "int",

                "description": (
                    "Execution timeout from 1 to 120 seconds."
                ),
            },
        },

        "result_formatter": (
            format_developer_execution_result
        ),

        "presentation_builder": (
            build_developer_execution_card
        ),

        "approval_formatter": (
            format_developer_execution_approval
        ),
    },

    "workspace_run_build": {
        "description": (
            "Run the trusted build/check command for a detected "
            "Python, Elixir, or Nim project."
        ),

        "risk":
            "medium",

        "requires_approval":
            True,

        "grounded_arguments":
            [],

        "parameters": {
            "relative_path": {
                "type":
                    "str",

                "description": (
                    "Optional workspace-relative project "
                    "directory. Defaults to the workspace root."
                ),
            },

            "timeout_seconds": {
                "type":
                    "int",

                "description": (
                    "Execution timeout from 1 to 120 seconds."
                ),
            },
        },

        "result_formatter": (
            format_developer_execution_result
        ),

        "presentation_builder": (
            build_developer_execution_card
        ),

        "approval_formatter": (
            format_developer_execution_approval
        ),
    },

    # ============================================================
    # RUNTIME / SERVICE INSPECTION
    # ============================================================

    "workspace_process_snapshot": {
        "description": (
            "Inspect a bounded host process snapshot containing "
            "process identifiers, state, runtime, and executable "
            "names without command lines or environment values."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments":
            [],

        "parameters":
            {},

        "result_formatter": (
            format_process_snapshot_result
        ),

        "presentation_builder": (
            build_process_snapshot_card
        ),
    },

    "workspace_service_status": {
        "description": (
            "Inspect declared workspace service status through "
            "the configured trusted runtime provider."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments":
            [],

        "parameters": {
            "relative_path": {
                "type":
                    "str",

                "description": (
                    "Optional workspace-relative directory "
                    "containing the service configuration."
                ),
            },
        },

        "result_formatter": (
            format_service_status_result
        ),

        "presentation_builder": (
            build_service_status_card
        ),
    },

    "workspace_service_logs": {
        "description": (
            "Read a bounded recent log tail for one declared "
            "workspace service with common secret patterns "
            "redacted."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "service_name",
        ],

        "parameters": {
            "service_name": {
                "type":
                    "str",

                "description": (
                    "Exact declared service name supplied "
                    "by the user."
                ),
            },

            "relative_path": {
                "type":
                    "str",

                "description": (
                    "Optional workspace-relative directory "
                    "containing the service configuration."
                ),
            },
        },

        "result_formatter": (
            format_service_logs_result
        ),

        "presentation_builder": (
            build_service_logs_card
        ),
    },
}