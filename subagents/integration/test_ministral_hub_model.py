from subagents.llm.ministral_hub import (
    MinistralHubBackend,
)


def test_clean_plain_json():
    response = (
        '{"agents": ["account-specialist"]}'
    )

    result = MinistralHubBackend._clean_response(
        response
    )

    assert result == response


def test_clean_json_markdown_fence():
    response = """
```json
{
  "agents": ["account-specialist"]
}
```
"""

def test_clean_response_removes_fence_and_eos():
    raw = (
        '```json\n'
        '{"agents": ["access-specialist"]}\n'
        '```</s>'
    )

    cleaned = (
        MinistralHubBackend
        ._clean_response(raw)
    )

    assert cleaned == (
        '{"agents": ["access-specialist"]}'
    )