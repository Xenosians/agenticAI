import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = Path(
    "/mnt/c/project/agenticaiPersonal/Models/qwen2.5-0.5b-funccall"
)


ACCOUNT_TOOLS = [
    {
        "name": "account_status",
        "description": (
            "Check whether a user account exists, is enabled, "
            "and whether it is locked."
        ),
        "parameters": {
            "user_id": {
                "description": "Exact user identifier to check.",
                "type": "str",
            }
        },
    },
    {
        "name": "unlock_user",
        "description": "Unlock a locked user account.",
        "parameters": {
            "user_id": {
                "description": "Exact user identifier to unlock.",
                "type": "str",
            }
        },
    },
    {
        "name": "reset_password",
        "description": "Request a password reset for a user account.",
        "parameters": {
            "user_id": {
                "description": "Exact user identifier.",
                "type": "str",
            }
        },
    },
]


def build_system_prompt() -> str:
    return (
        "You are a function-calling assistant. "
        "Given a user query and a list of available tools, "
        "respond with ONLY a JSON array of the function call(s) "
        "needed to fulfill the query. "
        "Each item must have 'name' and 'arguments' keys. "
        "Do not include explanation, markdown, or any text "
        "outside the raw JSON array.\n\n"
        "Available tools:\n"
        f"{json.dumps(ACCOUNT_TOOLS, indent=2)}"
    )


@pytest.fixture(scope="module")
def model_bundle():
    assert MODEL_PATH.exists(), (
        f"Model does not exist at {MODEL_PATH}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )

    model.eval()

    return tokenizer, model


def generate_tool_call(
    model_bundle,
    user_request: str,
) -> list[dict]:
    tokenizer, model = model_bundle

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(),
        },
        {
            "role": "user",
            "content": user_request,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[
        0,
        inputs["input_ids"].shape[1]:,
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    print(
        f"\nREQUEST: {user_request}"
        f"\nRAW MODEL OUTPUT: {response}"
    )

    parsed = json.loads(response)

    assert isinstance(parsed, list)
    assert len(parsed) >= 1

    return parsed


def test_account_status_selection(model_bundle):
    calls = generate_tool_call(
        model_bundle,
        "Is jdoe locked?",
    )

    call = calls[0]

    assert call["name"] == "account_status"
    assert call["arguments"]["user_id"] == "jdoe"


def test_unlock_selection(model_bundle):
    calls = generate_tool_call(
        model_bundle,
        "Unlock jdoe",
    )

    call = calls[0]

    assert call["name"] == "unlock_user"
    assert call["arguments"]["user_id"] == "jdoe"


def test_password_reset_selection(model_bundle):
    calls = generate_tool_call(
        model_bundle,
        "Reset the password for jdoe",
    )

    call = calls[0]

    assert call["name"] == "reset_password"
    assert call["arguments"]["user_id"] == "jdoe"