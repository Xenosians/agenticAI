from config import (
    Settings,
)

from services.jira import (
    build_jira_project_read_service,
)


def main(
) -> None:

    settings = (
        Settings()
    )

    service = (
        build_jira_project_read_service(
            settings
        )
    )

    print(
        "Jira J2A Project Preflight"
    )

    print(
        "=========================="
    )

    result = (
        service.list_projects(
            limit=25
        )
    )

    if not result.get(
        "ok"
    ):
        raise SystemExit(
            "Project discovery failed: "
            + str(
                result.get(
                    "error"
                )
            )
        )

    print()
    print(
        "Visible projects:"
    )

    for project in (
        result.get(
            "projects",
            []
        )
    ):
        print(
            "  "
            f"{project.get('key')} "
            f"- {project.get('name')} "
            f"- {project.get('project_type_key')}"
        )

    print()
    print(
        "Count:",
        result.get(
            "count"
        ),
    )

    print(
        "Total:",
        result.get(
            "total"
        ),
    )

    print(
        "Truncated:",
        result.get(
            "truncated"
        ),
    )

    projects = (
        result.get(
            "projects",
            []
        )
    )

    if projects:
        project_key = (
            projects[
                0
            ].get(
                "key"
            )
        )

        detail = (
            service.get_project(
                project_key
            )
        )

        if not detail.get(
            "ok"
        ):
            raise SystemExit(
                "Exact project lookup failed: "
                + str(
                    detail.get(
                        "error"
                    )
                )
            )

        print()
        print(
            "Exact lookup:"
        )

        print(
            "  key:",
            detail[
                "project"
            ].get(
                "key"
            ),
        )

        print(
            "  name:",
            detail[
                "project"
            ].get(
                "name"
            ),
        )

        print(
            "  id:",
            detail[
                "project"
            ].get(
                "id"
            ),
        )

    print()
    print(
        "J2A Jira project read-only preflight passed."
    )


if __name__ == "__main__":
    main()
