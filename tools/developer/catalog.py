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
from tools.workspace.catalog import (
    WORKSPACE_REPOSITORY_PARAMETER,
    resolve_workspace_argument_values,
)


DEVELOPER_TOOLS = {
    # ============================================================
    # PROJECT DETECTION / EXECUTION
    # ============================================================

    "workspace_project_info": {
        "description": (
            "Detect the supported project type and "
            "available governed developer operations "
            "inside a configured logical repository. Use repository "
            "for repository scope and relative_path only for a path inside it."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "repository",
        ],

        "argument_values_resolver":
            resolve_workspace_argument_values,

        "parameters": {
            "repository": WORKSPACE_REPOSITORY_PARAMETER,

            "relative_path": {
                "type":
                    "str",

                "description": (
                    "Optional project directory inside the selected "
                    "repository. Omit it for the repository root; "
                    "never send \".\" or a guessed path."
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
            "Python, Elixir, or Nim project inside a configured logical "
            "repository. Use repository for repository scope."
        ),

        "risk":
            "medium",

        "requires_approval":
            True,

        "grounded_arguments": [
            "repository",
        ],

        "argument_values_resolver":
            resolve_workspace_argument_values,

        "parameters": {
            "repository": WORKSPACE_REPOSITORY_PARAMETER,

            "relative_path": {
                "type":
                    "str",

                "description": (
                    "Optional project directory inside the selected "
                    "repository. Omit it for the repository root; "
                    "never send \".\" or a guessed path."
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
            "Python, Elixir, or Nim project inside a configured logical "
            "repository. Use repository for repository scope."
        ),

        "risk":
            "medium",

        "requires_approval":
            True,

        "grounded_arguments": [
            "repository",
        ],

        "argument_values_resolver":
            resolve_workspace_argument_values,

        "parameters": {
            "repository": WORKSPACE_REPOSITORY_PARAMETER,

            "relative_path": {
                "type":
                    "str",

                "description": (
                    "Optional project directory inside the selected "
                    "repository. Omit it for the repository root; "
                    "never send \".\" or a guessed path."
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

        "grounded_arguments": [
            "relative_path",
        ],

        "parameters": {
            "relative_path": {
                "type":
                    "str",

                "default": ".",

                "description": (
                    "Workspace-relative directory that contains the "
                    "Docker Compose file. Copy it exactly when the user "
                    "names one (for example projectOps/.runtime/mvp-workspace); "
                    "omit it only when the user names none."
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
            "relative_path",
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

                "default": ".",

                "description": (
                    "Workspace-relative directory that contains the "
                    "Docker Compose file. Copy it exactly when the user "
                    "names one (for example projectOps/.runtime/mvp-workspace); "
                    "omit it only when the user names none."
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
