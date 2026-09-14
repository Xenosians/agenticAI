from tools.developer_presentation import (
    format_developer_execution_approval,
    format_developer_execution_result,
    format_project_info_result,
)


DEVELOPER_TOOLS = {
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

        "approval_formatter": (
            format_developer_execution_approval
        ),
    },
}