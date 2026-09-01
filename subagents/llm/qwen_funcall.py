from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from subagents.llm.base import LLMBackend


class QwenFuncCallBackend(LLMBackend):
    """
    Backend for the Qwen2.5 0.5B function-calling worker.
    """

    def __init__(
        self,
        model_path: str | Path,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            fix_mistral_regex=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
        )

        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 128,
    ) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = output[
            0,
            inputs["input_ids"].shape[1]:,
        ]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()