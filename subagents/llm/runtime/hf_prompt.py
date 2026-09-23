from __future__ import annotations

import hashlib


HF_CAUSAL_FALLBACK_PROMPT_SCHEMA = (
    "hf-causal-role-labelled-prompt.v1|"
    "roles:system=SYSTEM,user=USER,assistant=ASSISTANT|"
    "message:{LABEL}:\\n{content.strip()}|"
    "separator:\\n\\n|"
    "generation_suffix:ASSISTANT:\\n"
)


def hf_causal_fallback_prompt_sha256(
) -> str:
    """
    Stable identity of the deterministic prompt renderer used by
    HFCausalWorkerBackend when a tokenizer has no native chat
    template.

    Bump HF_CAUSAL_FALLBACK_PROMPT_SCHEMA whenever the rendering
    contract changes semantically.
    """

    return (
        hashlib
        .sha256(
            HF_CAUSAL_FALLBACK_PROMPT_SCHEMA
            .encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


def resolve_prompt_renderer_sha256(
    *,
    backend: str,
    native_chat_template_sha256: (
        str
        | None
    ),
) -> (
    str
    | None
):
    """
    Resolve the effective prompt-renderer identity for one model
    profile.

    Native tokenizer chat templates retain their artifact hash.
    Generic hf-causal specialists without one use the explicit
    versioned fallback renderer identity.

    Other backends fail closed when no native chat-template hash is
    available because this module does not claim knowledge of their
    rendering behavior.
    """

    if (
        native_chat_template_sha256
        is not None
    ):
        return (
            native_chat_template_sha256
        )

    if (
        backend.strip().lower()
        == "hf-causal"
    ):
        return (
            hf_causal_fallback_prompt_sha256()
        )

    return None


def render_hf_causal_fallback_prompt(
    messages: list[
        dict[
            str,
            str,
        ]
    ],
) -> str:
    """
    Deterministic no-chat-template prompt rendering contract.

    This function and HF_CAUSAL_FALLBACK_PROMPT_SCHEMA are one
    versioned behavior. Change both together.
    """

    parts: list[str] = []

    role_labels = {
        "system":
            "SYSTEM",

        "user":
            "USER",

        "assistant":
            "ASSISTANT",
    }

    for message in messages:
        role = (
            message.get(
                "role",
                "user",
            )
        )

        content = (
            message.get(
                "content",
                ""
            )
        )

        label = (
            role_labels.get(
                role,
                role.upper(),
            )
        )

        parts.append(
            (
                f"{label}:\n"
                f"{content.strip()}"
            )
        )

    parts.append(
        "ASSISTANT:\n"
    )

    return (
        "\n\n".join(
            parts
        )
    )
