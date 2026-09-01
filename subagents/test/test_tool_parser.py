import pytest

from subagents.core.tool_parser import parse_tool_calls


def test_parse_single_tool_call():
    response = """
    [
        {
            "name": "account_status",
            "arguments": {
                "user_id": "jdoe"
            }
        }
    ]
    """

    calls = parse_tool_calls(response)

    assert calls == [
        {
            "name": "account_status",
            "arguments": {
                "user_id": "jdoe"
            },
        }
    ]


def test_parse_multiple_tool_calls():
    response = """
    [
        {
            "name": "account_status",
            "arguments": {
                "user_id": "jdoe"
            }
        },
        {
            "name": "account_status",
            "arguments": {
                "user_id": "asmith"
            }
        }
    ]
    """

    calls = parse_tool_calls(response)

    assert len(calls) == 2
    assert calls[0]["arguments"]["user_id"] == "jdoe"
    assert calls[1]["arguments"]["user_id"] == "asmith"


def test_invalid_json_raises_error():
    with pytest.raises(ValueError):
        parse_tool_calls(
            "account_status(jdoe)"
        )


def test_non_list_response_raises_error():
    with pytest.raises(ValueError):
        parse_tool_calls(
            '{"name": "account_status", "arguments": {}}'
        )


def test_missing_tool_name_raises_error():
    with pytest.raises(ValueError):
        parse_tool_calls(
            '[{"arguments": {"user_id": "jdoe"}}]'
        )


def test_invalid_arguments_raises_error():
    with pytest.raises(ValueError):
        parse_tool_calls(
            '[{"name": "account_status", "arguments": "jdoe"}]'
        )