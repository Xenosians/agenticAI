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