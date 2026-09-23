from services.ticketing import (
    JiraTicketService,
    MockTicketService,
    build_ticket_mutation_service,
)

from tools.registry import (
    get_tool,
)


def test_ticket_add_comment_catalog_requires_medium_risk_approval():
    tool = get_tool("ticket_add_comment")
    assert tool is not None
    assert tool["risk"] == "medium"
    assert tool["requires_approval"] is True
    assert tool["grounded_arguments"] == ["ticket_key", "comment"]
    assert "policy_resolver" in tool
    assert "expected_issue_id" in tool["trusted_policy_arguments"]


def test_mock_comment_mutation_is_visible_to_reads():
    service = MockTicketService()
    mutations = build_ticket_mutation_service(service)

    before = service.get_ticket_comments("ITSM-101", limit=20)
    assert before.count == 1

    result = mutations.add_comment(
        "ITSM-101",
        "VPN access was verified.",
    )

    assert result.ok is True
    assert result.status == "success"
    assert result.changed is True
    assert result.provider == "mock"
    assert result.ticket_key == "ITSM-101"

    after = service.get_ticket_comments("ITSM-101", limit=20)
    assert after.count == 2
    assert after.comments[-1].body == "VPN access was verified."


def test_mock_comment_mutation_rejects_empty_comment():
    service = MockTicketService()
    mutations = build_ticket_mutation_service(service)

    result = mutations.add_comment("ITSM-101", "   ")

    assert result.ok is False
    assert result.status == "denied"
    assert result.changed is False


def test_mock_comment_mutation_unknown_ticket_is_not_found():
    service = MockTicketService()
    mutations = build_ticket_mutation_service(service)

    result = mutations.add_comment(
        "ITSM-999",
        "Investigation started.",
    )

    assert result.ok is False
    assert result.status == "not_found"
    assert result.changed is False


def test_jira_comment_mutation_fails_closed_without_trusted_snapshot():
    calls = 0

    def forbidden_transport(request):
        nonlocal calls
        calls += 1
        raise AssertionError(
            f"No Jira request should execute without approval snapshot: {request.url}"
        )

    import httpx

    service = JiraTicketService(
        base_url="https://example.atlassian.net",
        email="agent@example.com",
        api_token="test-token",
        transport=httpx.MockTransport(forbidden_transport),
    )

    result = build_ticket_mutation_service(service).add_comment(
        "ITSM-101",
        "Investigation started.",
    )

    assert result.ok is False
    assert result.status == "denied"
    assert result.mutation_performed is False
    assert result.verification_ok is False
    assert calls == 0
    service.close()


def test_jira_assignment_and_transition_fail_closed_without_snapshots():
    calls = 0

    def forbidden_transport(request):
        nonlocal calls
        calls += 1
        raise AssertionError(
            f"No Jira request should execute without approval snapshot: {request.url}"
        )

    import httpx

    service = JiraTicketService(
        base_url="https://example.atlassian.net",
        email="agent@example.com",
        api_token="test-token",
        transport=httpx.MockTransport(forbidden_transport),
    )
    mutations = build_ticket_mutation_service(service)

    assigned = mutations.assign_ticket("ITSM-101", "Alice")
    transitioned = mutations.transition_ticket("ITSM-101", "Resolved")

    assert assigned.ok is False
    assert assigned.status == "denied"
    assert assigned.mutation_performed is False
    assert transitioned.ok is False
    assert transitioned.status == "denied"
    assert transitioned.mutation_performed is False
    assert calls == 0
    service.close()
