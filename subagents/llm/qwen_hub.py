import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from subagents.llm.base import LLMBackend


class QwenHubBackend(LLMBackend):
    """
    Qwen3-0.6B backend used by the hub/orchestrator.
    """

    def __init__(
        self,
        model_path: str | Path,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Hub model not found: {self.model_path}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
        )

        self.model.eval()

    def _clean_response(
        self,
        text: str,
    ) -> str:
        # Remove complete thinking blocks.
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Remove unfinished thinking block.
        if "<think>" in text.lower():
            index = text.lower().find("<think>")
            text = text[:index]

        return text.strip()

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 128,
    ) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[
            0,
            inputs["input_ids"].shape[1]:,
        ]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return self._clean_response(
            response
        )