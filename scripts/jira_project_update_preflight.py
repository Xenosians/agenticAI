from pprint import (
    pprint,
)

from config import (
    Settings,
)

from services.jira import (
    build_jira_project_mutation_service,
)


PROJECT = (
    "J2BTST"
)

NEW_NAME = (
    "J2C Rename Proof"
)


def main(
) -> None:

    service = (
        build_jira_project_mutation_service(
            Settings()
        )
    )

    result = (
        service.prepare_update_project(
            project_id_or_key=(
                PROJECT
            ),

            new_name=(
                NEW_NAME
            ),
        )
    )

    print(
        "Jira J2C.1 Rename Preflight"
    )

    print(
        "==========================="
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
            "new name:",
            repr(
                arguments.get(
                    "new_name"
                )
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
            "trusted current name:",
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
        "No Jira mutation was executed."
    )

    print(
        "J2C.1 trusted rename preparation passed."
    )


if __name__ == "__main__":
    main()
