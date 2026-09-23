import asyncio

from pathlib import (
    Path,
)

from types import (
    SimpleNamespace,
)

import httpx

import tools.ticketing.policy as ticketing_policy

from services.ticketing.jira import (
    JiraTicketService,
)

from subagents.core.definitions.loader import (
    load_agent_directory,
)

from subagents.core.tooling.gateway import (
    ToolGateway,
)

from tools.registry import (
    get_tool,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[4]
)

AGENTS_DIR = (
    PROJECT_ROOT
    / "subagents"
    / "agents"
)


def jira_agent():

    agents = {
        agent.name:
            agent

        for agent
        in load_agent_directory(
            AGENTS_DIR
        )
    }

    return (
        agents[
            "jira-specialist"
        ]
    )


def jira_service():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        if (
            request.method
            == "GET"
            and request.url.path
            == "/rest/api/3/project/KAN"
        ):
            return (
                httpx.Response(
                    200,
                    json={
                        "id":
                            "10001",

                        "key":
                            "KAN",

                        "name":
                            "Kanban",
                    },
                )
            )

        if (
            request.method
            == "GET"
            and request.url.path
            == (
                "/rest/api/3/issue/"
                "createmeta/10001/issuetypes"
            )
        ):
            return (
                httpx.Response(
                    200,
                    json={
                        "issueTypes": [
                            {
                                "id":
                                    "10010",

                                "name":
                                    "Task",

                                "subtask":
                                    False,
                            },
                        ],

                        "startAt":
                            0,

                        "maxResults":
                            50,

                        "total":
                            1,
                    },
                )
            )

        raise AssertionError(
            (
                "Unexpected provider request: "
                f"{request.method} "
                f"{request.url}"
            )
        )

    return (
        JiraTicketService(
            base_url=(
                "https://example.atlassian.net"
            ),

            email=(
                "agent@example.com"
            ),

            api_token=(
                "test-token"
            ),

            transport=(
                httpx.MockTransport(
                    handler
                )
            ),
        )
    )


def test_ticket_create_catalog_uses_medium_risk_policy():

    tool = (
        get_tool(
            "ticket_create"
        )
    )

    assert tool is not None

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
            "policy_resolver"
        ]
        is (
            ticketing_policy
            .evaluate_ticket_create_policy
        )
    )

    assert (
        tool[
            "trusted_policy_arguments"
        ]
        == [
            "expected_project_id",
            "expected_project_key",
            "expected_project_name",
            "expected_ticket_type_id",
            "expected_ticket_type_name",
        ]
    )


def test_non_jira_policy_preserves_exact_model_arguments(
    monkeypatch,
):

    monkeypatch.setattr(
        ticketing_policy,
        "Settings",
        lambda: (
            SimpleNamespace(
                ticketing_backend=(
                    "mock"
                )
            )
        ),
    )

    result = (
        ticketing_policy
        .evaluate_ticket_create_policy(
            project_key="ITSM",
            summary=(
                "Laptop onboarding failure"
            ),
            ticket_type="Task",
        )
    )

    assert result == {
        "ok":
            True,

        "status":
            "ready",

        "risk":
            "medium",

        "requires_approval":
            True,

        "execution_arguments": {
            "project_key":
                "ITSM",

            "summary":
                "Laptop onboarding failure",

            "ticket_type":
                "Task",
        },

        "error":
            None,
    }


def test_jira_policy_binds_provider_snapshot_and_closes_service(
    monkeypatch,
):

    service = (
        jira_service()
    )

    monkeypatch.setattr(
        ticketing_policy,
        "Settings",
        lambda: (
            SimpleNamespace(
                ticketing_backend=(
                    "jira"
                )
            )
        ),
    )

    monkeypatch.setattr(
        ticketing_policy,
        "build_ticket_service",
        lambda _settings: (
            service
        ),
    )

    result = (
        ticketing_policy
        .evaluate_ticket_create_policy(
            project_key="KAN",
            summary="VPN login failure",
        )
    )

    assert result["ok"] is True
    assert result["status"] == "ready"
    assert result["risk"] == "medium"

    assert (
        result[
            "execution_arguments"
        ]
        == {
            "project_key":
                "KAN",

            "summary":
                "VPN login failure",

            "expected_project_id":
                "10001",

            "expected_project_key":
                "KAN",

            "expected_project_name":
                "Kanban",

            "expected_ticket_type_id":
                "10010",

            "expected_ticket_type_name":
                "Task",
        }
    )

    assert (
        service.client.is_closed
        is True
    )


def test_gateway_persists_exact_jira_create_snapshot_for_approval(
    monkeypatch,
):

    created_services = []

    def build_service(
        _settings,
    ):

        service = (
            jira_service()
        )

        created_services.append(
            service
        )

        return service

    monkeypatch.setattr(
        ticketing_policy,
        "Settings",
        lambda: (
            SimpleNamespace(
                ticketing_backend=(
                    "jira"
                )
            )
        ),
    )

    monkeypatch.setattr(
        ticketing_policy,
        "build_ticket_service",
        build_service,
    )

    approvals = []

    def create_approval(
        tool_name,
        arguments,
        risk=None,
    ):

        approvals.append(
            {
                "tool":
                    tool_name,

                "arguments":
                    arguments,

                "risk":
                    risk,
            }
        )

        return {
            "id":
                "approval-test",
        }

    class NeverExecuteMCP:

        async def call_tool(
            self,
            *_args,
            **_kwargs,
        ):
            raise AssertionError(
                "MCP execution must not occur "
                "during approval preparation."
            )

    gateway = (
        ToolGateway(
            approval_creator=(
                create_approval
            ),

            mcp=(
                NeverExecuteMCP()
            ),
        )
    )

    async def scenario():

        return (
            await gateway.execute(
                agent=(
                    jira_agent()
                ),

                user_input=(
                    "Create a Jira ticket in KAN "
                    "titled VPN login failure."
                ),

                tool_name=(
                    "ticket_create"
                ),

                arguments={
                    "project_key":
                        "KAN",

                    "summary":
                        "VPN login failure",
                },
            )
        )

    result = (
        asyncio.run(
            scenario()
        )
    )

    assert result["ok"] is True

    assert (
        result[
            "status"
        ]
        == "approval_required"
    )

    assert (
        result[
            "risk"
        ]
        == "medium"
    )

    assert (
        result[
            "approval_id"
        ]
        == "approval-test"
    )

    assert (
        result[
            "approval_arguments"
        ]
        == {
            "project_key":
                "KAN",

            "summary":
                "VPN login failure",

            "expected_project_id":
                "10001",

            "expected_project_key":
                "KAN",

            "expected_project_name":
                "Kanban",

            "expected_ticket_type_id":
                "10010",

            "expected_ticket_type_name":
                "Task",
        }
    )

    assert approvals == [
        {
            "tool":
                "ticket_create",

            "arguments":
                {
                    "project_key":
                        "KAN",

                    "summary":
                        "VPN login failure",

                    "expected_project_id":
                        "10001",

                    "expected_project_key":
                        "KAN",

                    "expected_project_name":
                        "Kanban",

                    "expected_ticket_type_id":
                        "10010",

                    "expected_ticket_type_name":
                        "Task",
                },

            "risk":
                "medium",
        }
    ]

    assert len(
        created_services
    ) == 1

    assert (
        created_services[
            0
        ]
        .client
        .is_closed
        is True
    )
