import tools.jira.mcp as jira_mcp

from tools.jira.catalog import (
    JIRA_PROJECT_CREATE_TRUSTED_ARGUMENTS,
    JIRA_TOOLS,
)


class FakeMCPServer:

    def __init__(
        self,
    ) -> None:

        self.functions = {}

    def tool(
        self,
    ):

        def decorator(
            function,
        ):

            self.functions[
                function.__name__
            ] = (
                function
            )

            return (
                function
            )

        return (
            decorator
        )


class FakeReadService:

    def list_projects(
        self,
        *,
        query=None,
        limit=25,
    ):

        return {
            "ok":
                True,

            "status":
                "success",

            "query":
                query,

            "projects": [
                {
                    "id":
                        "10001",

                    "key":
                        "OPS",

                    "name":
                        "Operations",

                    "project_type_key":
                        "software",

                    "style":
                        "classic",

                    "simplified":
                        False,

                    "archived":
                        False,

                    "category_id":
                        None,

                    "category_name":
                        None,
                }
            ],

            "count":
                1,

            "total":
                1,

            "truncated":
                False,

            "error":
                None,
        }

    def get_project(
        self,
        project_id_or_key,
    ):

        return {
            "ok":
                True,

            "status":
                "success",

            "project_id_or_key":
                project_id_or_key,

            "project": {
                "id":
                    "10001",

                "key":
                    project_id_or_key,

                "name":
                    "Operations",

                "project_type_key":
                    "software",

                "style":
                    "classic",

                "simplified":
                    False,

                "archived":
                    False,

                "category_id":
                    None,

                "category_name":
                    None,
            },

            "error":
                None,
        }


class FakeMutationService:

    def create_project(
        self,
        *,
        project_key,
        project_name,
        template,
        expected_project_type_key,
        expected_project_template_key,
        expected_lead_account_id,
    ):

        return {
            "ok":
                True,

            "status":
                "success",

            "operation":
                "create",

            "project_id":
                "10002",

            "project_key":
                project_key,

            "project_name":
                project_name,

            "template":
                template,

            "project_type_key":
                expected_project_type_key,

            "mutation_performed":
                True,

            "verification_ok":
                True,

            "reconciled":
                False,

            "error":
                None,
        }


def test_j2a_capabilities_remain_read_only():

    for tool_name in (
        "jira_project_list",
        "jira_project_get",
    ):

        tool = (
            JIRA_TOOLS[
                tool_name
            ]
        )

        assert (
            tool[
                "risk"
            ]
            == "read"
        )

        assert (
            tool[
                "requires_approval"
            ]
            is False
        )


def test_project_get_is_exactly_grounded():

    assert (
        JIRA_TOOLS[
            "jira_project_get"
        ][
            "grounded_arguments"
        ]
        == [
            "project_id_or_key",
        ]
    )


def test_j2b_project_create_is_governed():

    tool = (
        JIRA_TOOLS[
            "jira_project_create"
        ]
    )

    assert (
        tool[
            "risk"
        ]
        == "medium"
    )

    assert (
        tool[
            "requires_approval"
        ]
        is True
    )

    assert (
        tool[
            "policy_owns_preconditions"
        ]
        is True
    )

    assert (
        tool[
            "grounded_arguments"
        ]
        == [
            "project_key",
            "project_name",
            "template",
        ]
    )

    assert (
        tool[
            "trusted_policy_arguments"
        ]
        == (
            JIRA_PROJECT_CREATE_TRUSTED_ARGUMENTS
        )
    )

    assert (
        set(
            tool[
                "parameters"
            ]
        )
        == {
            "project_key",
            "project_name",
            "template",
        }
    )


def test_jira_template_aliases_are_bounded():

    resolver = (
        JIRA_TOOLS[
            "jira_project_create"
        ][
            "argument_values_resolver"
        ]
    )

    result = (
        resolver()
    )

    assert (
        result
        == {
            "template": [
                "software-kanban",
                "software-scrum",
                "business-task-tracking",
                "service-it-management",
            ],
        }
    )


def test_all_jira_capabilities_register_with_mcp():

    server = (
        FakeMCPServer()
    )

    jira_mcp.register_jira_project_tools(
        server,
        FakeReadService(),
        FakeMutationService(),
    )

    assert (
        set(
            server.functions
        )
        == set(
            JIRA_TOOLS
        )
    )


def test_jira_project_create_mcp_preserves_approved_snapshot():

    server = (
        FakeMCPServer()
    )

    jira_mcp.register_jira_project_tools(
        server,
        FakeReadService(),
        FakeMutationService(),
    )

    result = (
        server.functions[
            "jira_project_create"
        ](
            project_key="NEW",
            project_name="New Project",
            template="software-kanban",
            expected_project_type_key="software",
            expected_project_template_key="trusted-template",
            expected_lead_account_id="account-1",
        )
    )

    assert (
        result.ok
        is True
    )

    assert (
        result.project_key
        == "NEW"
    )

    assert (
        result.verification_ok
        is True
    )

    assert (
        result.mutation_performed
        is True
    )


def test_j2c2_project_archive_is_high_risk_governed():

    tool = (
        JIRA_TOOLS[
            "jira_project_archive"
        ]
    )

    assert (
        tool[
            "risk"
        ]
        == "high"
    )

    assert (
        tool[
            "requires_approval"
        ]
        is True
    )

    assert (
        tool[
            "policy_owns_preconditions"
        ]
        is True
    )

    assert (
        tool[
            "grounded_arguments"
        ]
        == [
            "project_id_or_key",
        ]
    )

    assert (
        set(
            tool[
                "parameters"
            ]
        )
        == {
            "project_id_or_key",
        }
    )

    assert (
        set(
            tool[
                "trusted_policy_arguments"
            ]
        )
        == {
            "expected_project_id",
            "expected_project_key",
            "expected_project_name",
        }
    )


def test_archive_mcp_preserves_archived_result_field():

    class ArchiveMutationService:

        def archive_project(
            self,
            *,
            project_id_or_key,
            expected_project_id,
            expected_project_key,
            expected_project_name,
        ):

            return {
                "ok":
                    True,

                "status":
                    "success",

                "operation":
                    "archive",

                "project_id":
                    expected_project_id,

                "project_key":
                    expected_project_key,

                "project_name":
                    expected_project_name,

                "previous_project_name":
                    None,

                "new_project_name":
                    None,

                "template":
                    None,

                "project_type_key":
                    None,

                "archived":
                    True,

                "mutation_performed":
                    True,

                "verification_ok":
                    True,

                "reconciled":
                    False,

                "error":
                    None,
            }

    server = (
        FakeMCPServer()
    )

    jira_mcp.register_jira_project_tools(
        server,
        FakeReadService(),
        ArchiveMutationService(),
    )

    result = (
        server.functions[
            "jira_project_archive"
        ](
            project_id_or_key="J2BTST",
            expected_project_id="10033",
            expected_project_key="J2BTST",
            expected_project_name="J2C Rename Proof",
        )
    )

    assert (
        result.ok
        is True
    )

    assert (
        result.status
        == "success"
    )

    assert (
        result.operation
        == "archive"
    )

    assert (
        result.archived
        is True
    )

    assert (
        result.mutation_performed
        is True
    )

    assert (
        result.verification_ok
        is True
    )


def test_mutation_result_schema_contains_archive_truth():

    fields = (
        jira_mcp
        .JiraProjectMutationResult
        .model_fields
    )

    assert (
        "archived"
        in fields
    )
