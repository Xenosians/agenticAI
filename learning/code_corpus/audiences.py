from __future__ import annotations


def classify_chunk_audiences(
    *,
    source_kind: str,
    relative_path: str,
) -> list[str]:
    """
    Decide which model component owns this corpus material.

    Important separation:

    GENERAL corpus
        -> Hub/general model only.

    PROJECT corpus
        -> developer-specialist by default;
        -> relevant domain specialists additionally receive their narrow code.

    Project-specific implementation code is deliberately NOT sent to the Hub
    by this classifier.
    """

    if source_kind == "general":
        return [
            "hub",
        ]

    if source_kind != "project":
        raise ValueError(
            f"Unsupported source_kind: {source_kind}"
        )

    path = (
        relative_path
        .replace("\\", "/")
        .casefold()
    )

    audiences = [
        "developer-specialist",
    ]

    rules = [
        (
            "jira-specialist",
            (
                "tools/jira/",
                "services/jira/",
                "/jira_",
                "jira/",
            ),
        ),
        (
            "ticket-specialist",
            (
                "tools/ticketing/",
                "services/ticketing/",
                "ticketing/",
            ),
        ),
        (
            "account-specialist",
            (
                "tools/account/",
                "services/account",
                "account-specialist",
                "/ldap",
                "ldap_",
            ),
        ),
        (
            "access-specialist",
            (
                "tools/access/",
                "services/access",
                "access-specialist",
            ),
        ),
        (
            "asset-specialist",
            (
                "tools/asset/",
                "services/asset",
                "asset-specialist",
            ),
        ),
        (
            "knowledge-specialist",
            (
                "tools/knowledge/",
                "services/knowledge",
                "knowledge-specialist",
                "runbook",
            ),
        ),
        (
            "atlassian-specialist",
            (
                "tools/atlassian/",
                "services/atlassian",
                "atlassian-specialist",
            ),
        ),
    ]

    for audience, markers in rules:
        if any(
            marker in path
            for marker in markers
        ):
            audiences.append(
                audience
            )

    # Provider-neutral ticket capabilities are used by the Jira specialist
    # as well as the provider-neutral ticket specialist.
    if (
        "ticket-specialist"
        in audiences
        and "jira-specialist"
        not in audiences
    ):
        audiences.append(
            "jira-specialist"
        )

    return list(
        dict.fromkeys(
            audiences
        )
    )


def curriculum_hints_for_chunk(
    *,
    audiences: list[str],
    symbol_kind: str,
) -> list[str]:
    hints: list[str] = []

    if "developer-specialist" in audiences:
        hints.append(
            "dev-04-code-symbols"
        )

        if symbol_kind in {
            "test",
            "test_function",
            "test_module",
        }:
            hints.append(
                "dev-03-build-test"
            )

    return hints
