import argparse

from config import (
    Settings,
)

from services.jira import (
    build_jira_project_mutation_service,
)


def main(
) -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Read-only Jira J2B project-create "
                "policy preflight."
            )
        )
    )

    parser.add_argument(
        "--key",
        required=True,
    )

    parser.add_argument(
        "--name",
        required=True,
    )

    parser.add_argument(
        "--template",
        required=True,
        choices=[
            "software-kanban",
            "software-scrum",
            "business-task-tracking",
            "service-it-management",
        ],
    )

    args = (
        parser.parse_args()
    )

    service = (
        build_jira_project_mutation_service(
            Settings()
        )
    )

    result = (
        service.prepare_create_project(
            project_key=(
                args.key
            ),

            project_name=(
                args.name
            ),

            template=(
                args.template
            ),
        )
    )

    print(
        "Jira J2B Create Preflight"
    )

    print(
        "========================="
    )

    print(
        "ok:",
        result.get(
            "ok"
        ),
    )

    print(
        "status:",
        result.get(
            "status"
        ),
    )

    print(
        "risk:",
        result.get(
            "risk"
        ),
    )

    print(
        "requires approval:",
        result.get(
            "requires_approval"
        ),
    )

    execution = (
        result.get(
            "execution_arguments"
        )
    )

    if isinstance(
        execution,
        dict,
    ):

        print(
            "project key:",
            execution.get(
                "project_key"
            ),
        )

        print(
            "project name:",
            execution.get(
                "project_name"
            ),
        )

        print(
            "template alias:",
            execution.get(
                "template"
            ),
        )

        print(
            "trusted project type bound:",
            bool(
                execution.get(
                    "expected_project_type_key"
                )
            ),
        )

        print(
            "trusted native template bound:",
            bool(
                execution.get(
                    "expected_project_template_key"
                )
            ),
        )

        # Deliberately do not print the account ID itself.
        print(
            "trusted project lead bound:",
            bool(
                execution.get(
                    "expected_lead_account_id"
                )
            ),
        )

    if not result.get(
        "ok"
    ):
        print(
            "error:",
            result.get(
                "error"
            ),
        )

        raise SystemExit(
            1
        )

    print()
    print(
        "No Jira mutation was executed."
    )

    print(
        "J2B trusted create preparation passed."
    )


if __name__ == "__main__":
    main()
