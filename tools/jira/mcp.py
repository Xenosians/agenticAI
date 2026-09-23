from pydantic import (
    BaseModel,
    Field,
)

from mcp.server import (
    MCPServer,
)

from services.jira import (
    JiraProjectMutationProvider,
    JiraProjectReadProvider,
)


class JiraProjectResult(
    BaseModel
):
    id: str
    key: str
    name: str

    project_type_key: str | None = None
    style: str | None = None
    simplified: bool | None = None
    archived: bool | None = None
    category_id: str | None = None
    category_name: str | None = None


class JiraProjectListResult(
    BaseModel
):
    ok: bool
    status: str
    query: str | None = None

    projects: list[
        JiraProjectResult
    ] = Field(
        default_factory=list
    )

    count: int = 0
    total: int | None = None
    truncated: bool = False
    error: str | None = None


class JiraProjectLookupResult(
    BaseModel
):
    ok: bool
    status: str
    project_id_or_key: str | None = None
    project: JiraProjectResult | None = None
    error: str | None = None


class JiraProjectMutationResult(
    BaseModel
):
    ok: bool
    status: str

    operation: str | None = None

    project_id: str | None = None
    project_key: str | None = None
    project_name: str | None = None

    previous_project_name: str | None = None
    new_project_name: str | None = None

    template: str | None = None
    project_type_key: str | None = None

    archived: bool | None = None
    deleted: bool | None = None

    mutation_performed: bool | None = None
    verification_ok: bool = False
    reconciled: bool = False

    error: str | None = None


def register_jira_project_tools(
    server: MCPServer,
    read_service: JiraProjectReadProvider,
    mutation_service: JiraProjectMutationProvider,
) -> None:

    @server.tool()
    def jira_project_list(
        query: str | None = None,
        limit: int | None = None,
    ) -> JiraProjectListResult:

        return (
            JiraProjectListResult(
                **read_service.list_projects(
                    query=(
                        query
                    ),

                    limit=(
                        25
                        if limit is None
                        else limit
                    ),
                )
            )
        )

    @server.tool()
    def jira_project_get(
        project_id_or_key: str,
    ) -> JiraProjectLookupResult:

        return (
            JiraProjectLookupResult(
                **read_service.get_project(
                    project_id_or_key
                )
            )
        )

    @server.tool()
    def jira_project_create(
        project_key: str,
        project_name: str,
        template: str,
        expected_project_type_key: str,
        expected_project_template_key: str,
        expected_lead_account_id: str,
    ) -> JiraProjectMutationResult:

        return (
            JiraProjectMutationResult(
                **mutation_service.create_project(
                    project_key=(
                        project_key
                    ),

                    project_name=(
                        project_name
                    ),

                    template=(
                        template
                    ),

                    expected_project_type_key=(
                        expected_project_type_key
                    ),

                    expected_project_template_key=(
                        expected_project_template_key
                    ),

                    expected_lead_account_id=(
                        expected_lead_account_id
                    ),
                )
            )
        )

    @server.tool()
    def jira_project_update(
        project_id_or_key: str,
        new_name: str,
        expected_project_id: str,
        expected_project_key: str,
        expected_project_name: str,
    ) -> JiraProjectMutationResult:

        return (
            JiraProjectMutationResult(
                **mutation_service.update_project(
                    project_id_or_key=(
                        project_id_or_key
                    ),

                    new_name=(
                        new_name
                    ),

                    expected_project_id=(
                        expected_project_id
                    ),

                    expected_project_key=(
                        expected_project_key
                    ),

                    expected_project_name=(
                        expected_project_name
                    ),
                )
            )
        )

    @server.tool()
    def jira_project_archive(
        project_id_or_key: str,
        expected_project_id: str,
        expected_project_key: str,
        expected_project_name: str,
    ) -> JiraProjectMutationResult:

        return (
            JiraProjectMutationResult(
                **mutation_service.archive_project(
                    project_id_or_key=(
                        project_id_or_key
                    ),

                    expected_project_id=(
                        expected_project_id
                    ),

                    expected_project_key=(
                        expected_project_key
                    ),

                    expected_project_name=(
                        expected_project_name
                    ),
                )
            )
        )


    @server.tool()
    def jira_project_delete(
        project_id_or_key: str,
        expected_project_id: str,
        expected_project_key: str,
        expected_project_name: str,
    ) -> JiraProjectMutationResult:

        return (
            JiraProjectMutationResult(
                **mutation_service.delete_project(
                    project_id_or_key=(
                        project_id_or_key
                    ),

                    expected_project_id=(
                        expected_project_id
                    ),

                    expected_project_key=(
                        expected_project_key
                    ),

                    expected_project_name=(
                        expected_project_name
                    ),
                )
            )
        )
