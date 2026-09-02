from subagents.llm.ministral_hub import MinistralHubBackend


def test_clean_response_plain_json():
    raw = '{"agents": ["account-specialist"]}'

    cleaned = (
        MinistralHubBackend
        ._clean_response(raw)
    )

    assert cleaned == (
        '{"agents": ["account-specialist"]}'
    )


def test_clean_response_removes_markdown_fence():
    raw = (
        '```json\n'
        '{"agents": ["account-specialist"]}\n'
        '```'
    )

    cleaned = (
        MinistralHubBackend
        ._clean_response(raw)
    )

    assert cleaned == (
        '{"agents": ["account-specialist"]}'
    )


def test_clean_response_removes_opening_fence():
    raw = (
        '```json\n'
        '{"agents": ["access-specialist"]}'
    )

    cleaned = (
        MinistralHubBackend
        ._clean_response(raw)
    )

    assert cleaned == (
        '{"agents": ["access-specialist"]}'
    )


def test_clean_response_removes_eos_token():
    raw = (
        '{"agents": ["access-specialist"]}</s>'
    )

    cleaned = (
        MinistralHubBackend
        ._clean_response(raw)
    )

    assert cleaned == (
        '{"agents": ["access-specialist"]}'
    )


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


def test_clean_response_removes_multiple_eos_tokens():
    raw = (
        '{"agents": []}</s></s>'
    )

    cleaned = (
        MinistralHubBackend
        ._clean_response(raw)
    )

    assert cleaned == '{"agents": []}'