from learning.evidence.hub_routing import (
    capture_hub_routing_generation,
    reset_hub_routing_trace,
)


def _capture(messages):
    reset_hub_routing_trace()

    return capture_hub_routing_generation(
        model_key="hub-main",
        model_profile_resolver=None,
        specialists=[
            {
                "name": "jira-specialist",
            }
        ],
        messages=messages,
        context_messages=[],
        user_request="Create a task.",
        repair_mode=False,
        include_workflow_protocol=True,
        repair_error=None,
        max_new_tokens=512,
        response='{"delegations":[]}',
    )


def test_clean_hub_messages_are_archived_exactly():
    messages = [
        {
            "role": "system",
            "content": "router prompt",
        },
        {
            "role": "user",
            "content": "Create a task.",
        },
    ]

    attempt = _capture(
        messages
    )

    assert attempt.messages_exact is True
    assert attempt.messages == messages
    assert attempt.sanitized_messages_sha256 is not None
    assert (
        attempt.sanitized_messages_sha256
        == attempt.messages_sha256
    )


def test_sensitive_hub_messages_fail_closed_for_exact_training_prompt():
    messages = [
        {
            "role": "system",
            "content": "router prompt",
        },
        {
            "role": "user",
            "content": "password=hunter2",
        },
    ]

    attempt = _capture(
        messages
    )

    assert attempt.messages_exact is False
    assert attempt.messages is not None
    assert "hunter2" not in str(
        attempt.messages
    )
