from config import (
    Settings,
)

from services.jira import (
    build_jira_project_mutation_service,
)


PROJECT = (
    "J2BTST"
)


def main(
) -> None:

    service = (
        build_jira_project_mutation_service(
            Settings()
        )
    )

    result = (
        service.prepare_archive_project(
            project_id_or_key=(
                PROJECT
            )
        )
    )

    print(
        "Jira J2C.2 Archive Preflight"
    )

    print(
        "============================"
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

    arguments = (
        result.get(
            "execution_arguments"
        )
    )

    if isinstance(
        arguments,
        dict,
    ):

        print()
        print(
            "project reference:",
            arguments.get(
                "project_id_or_key"
            ),
        )

        print(
            "trusted project ID:",
            arguments.get(
                "expected_project_id"
            ),
        )

        print(
            "trusted project key:",
            arguments.get(
                "expected_project_key"
            ),
        )

        print(
            "trusted project name:",
            repr(
                arguments.get(
                    "expected_project_name"
                )
            ),
        )

    if not result.get(
        "ok"
    ):

        print()
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
        "Trusted lifecycle state: live"
    )

    print(
        "No Jira mutation was executed."
    )

    print(
        "J2C.2 trusted archive preparation passed."
    )


if __name__ == "__main__":
    main()
