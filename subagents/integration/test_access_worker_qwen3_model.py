import json

import pytest

from config.settings import get_settings
from subagents.llm.qwen_hub import QwenHubBackend


SYSTEM_PROMPT = """
You are the Access Specialist in an ITSM agent system.

Your ONLY responsibility is proposing read-only access checks.

Available tool:

check_access

Arguments:
- user_id: exact user identifier explicitly present in the request
- resource: exact resource explicitly present in the request

SUPPORTED requests:
- asking whether a user can access a resource
- asking whether a user has access to a resource
- asking to check a user's access to a resource

UNSUPPORTED requests include:
- unlock account
- reset password
- account status
- changing permissions
- granting access
- revoking access
- unrelated conversation

CRITICAL RULES:

1. Use ONLY the check_access tool.
2. Never answer the access question yourself.
3. Never invent a user_id.
4. Never invent a resource.
5. Both user_id AND resource must be explicitly present in the request.
6. If either user_id or resource is missing, return [].
7. If the request is outside the access-check domain, return [].
8. Never convert an unsupported request into an access check.
9. Return ONLY JSON.
10. Do not use Markdown or ``` fences.
11. Produce exactly one tool call for a valid access request.

Valid response structure:

[
  {
    "name": "check_access",
    "arguments": {
      "user_id": "<exact user identifier>",
      "resource": "<exact resource identifier>"
    }
  }
]

The strings inside angle brackets above describe fields.
Do NOT literally output the angle brackets.

Unsupported request response:

[]

Example unsupported request:

User:
Reset the password for bob.

Assistant:
[]
""".strip()


CASES = [
    (
        "Does jdoe have VPN access?",
        "jdoe",
        "VPN",
    ),
    (
        "Can jdoe access VPN?",
        "jdoe",
        "VPN",
    ),
    (
        "Check VPN access for jdoe.",
        "jdoe",
        "VPN",
    ),
    (
        "Does alice have access to Finance?",
        "alice",
        "Finance",
    ),
]


def clean_json_response(raw: str) -> str:
    """
    Normalize harmless Markdown fencing from model output.

    Qwen3 sometimes emits:

        ```json
        [...]
        ```

    or even an opening fence without the closing fence.

    This function removes only the Markdown wrapper.
    It does NOT repair or invent JSON.
    """

    text = raw.strip()

    if text.startswith("```"):
        newline_index = text.find("\n")

        if newline_index == -1:
            return ""

        text = text[newline_index + 1 :].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


@pytest.fixture(scope="module")
def backend():
    settings = get_settings()

    model_path = settings.require_path(
        settings.access_model_path,
        "ACCESS_MODEL_PATH",
    )

    print()
    print("Loading Qwen3 Access candidate:")
    print(model_path)

    return QwenHubBackend(
        model_path=model_path,
    )


@pytest.mark.parametrize(
    "user_request,user_id,resource",
    CASES,
)
def test_qwen3_access_tool_call(
    backend,
    user_request,
    user_id,
    resource,
):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_request,
        },
    ]

    raw = backend.generate(
        messages,
        max_new_tokens=128,
    )

    cleaned = clean_json_response(raw)

    print()
    print("===== QWEN3 ACCESS =====")
    print("USER:", user_request)
    print("RAW:", raw)
    print("CLEANED:", cleaned)
    print("========================")

    payload = json.loads(cleaned)

    assert isinstance(payload, list)
    assert len(payload) == 1

    tool_call = payload[0]

    assert tool_call["name"] == "check_access"

    assert tool_call["arguments"] == {
        "user_id": user_id,
        "resource": resource,
    }


@pytest.mark.parametrize(
    "user_request",
    [
        "Unlock jdoe.",
        "Reset the password for jdoe.",
        "Is jdoe locked?",
        "Hello, how are you?",
    ],
)
def test_qwen3_access_rejects_out_of_domain(
    backend,
    user_request,
):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_request,
        },
    ]

    raw = backend.generate(
        messages,
        max_new_tokens=128,
    )

    cleaned = clean_json_response(raw)

    print()
    print("===== QWEN3 ACCESS OOD =====")
    print("USER:", user_request)
    print("RAW:", raw)
    print("CLEANED:", cleaned)
    print("============================")

    payload = json.loads(cleaned)

    assert payload == []