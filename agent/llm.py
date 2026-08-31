import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/mnt/c/project/agenticaiPersonal/Qwen3-0.6B"

THINK_PATTERN = re.compile(
    r"<think>.*?</think>",
    flags=re.DOTALL | re.IGNORECASE,
)


print("Loading Qwen...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
)

print("Qwen loaded.")


def _clean_response(text: str) -> str:
    # Remove complete thinking blocks.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove an unfinished thinking block.
    if "<think>" in text.lower():
        think_index = text.lower().find("<think>")
        text = text[:think_index]

    return text.strip()


def ask(messages: list[dict]) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return _clean_response(response)