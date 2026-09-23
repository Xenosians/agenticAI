from subagents.llm.runtime.hf_prompt import (
    hf_causal_fallback_prompt_sha256,
    render_hf_causal_fallback_prompt,
    resolve_prompt_renderer_sha256,
)


def test_hf_causal_fallback_renderer_is_deterministic_and_fingerprinted():
    messages = [
        {
            "role": "system",
            "content": " system ",
        },
        {
            "role": "user",
            "content": " hello ",
        },
    ]

    assert (
        render_hf_causal_fallback_prompt(
            messages
        )
        == (
            "SYSTEM:\n"
            "system\n\n"
            "USER:\n"
            "hello\n\n"
            "ASSISTANT:\n"
        )
    )

    fingerprint = (
        hf_causal_fallback_prompt_sha256()
    )

    assert len(
        fingerprint
    ) == 64

    assert (
        resolve_prompt_renderer_sha256(
            backend="hf-causal",
            native_chat_template_sha256=None,
        )
        == fingerprint
    )

    assert (
        resolve_prompt_renderer_sha256(
            backend="qwen3",
            native_chat_template_sha256="native-hash",
        )
        == "native-hash"
    )

    assert (
        resolve_prompt_renderer_sha256(
            backend="qwen3",
            native_chat_template_sha256=None,
        )
        is None
    )
