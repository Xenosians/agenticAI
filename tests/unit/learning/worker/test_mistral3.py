"""The shared worker trains a Mistral3 specialist base (Ministral 3B, the hub's base) like any other model.

Measured 19 Nov 2026 on the real checkpoint: it is ``mistral3`` (the Mistral3 conditional-generation class,
with a vision tower), which ``AutoModelForCausalLM`` does not map; its tokenizer is ``MistralCommonBackend`` with no
Jinja template, and rendering a prompt to text then tokenizing the text gave other ids (``[INST]`` split into
pieces: 33 ids instead of the 14 serving uses). These tests use a tiny Mistral3 built from the real
configuration's shape, with the real tokenizer files when they are on this machine.
"""
import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("trl")
pytest.importorskip("mistral_common")

from learning.worker.job import TrainingJob  # noqa: E402
from learning.worker.stages import run  # noqa: E402

REAL = Path("/mnt/c/project/agenticaiPersonal/models/Ministral-3-3B-Instruct-2512-BF16")
SYSTEM = "Answer with one tool call."
ANSWER = '[{"arguments":{},"name":"ping"}]'


@pytest.fixture(scope="module")
def tiny_mistral3(tmp_path_factory):
    if not (REAL / "tekken.json").is_file():
        pytest.skip("the Ministral tokenizer files are not on this machine")
    import transformers
    from transformers import AutoConfig, Mistral3Config

    from learning.worker.loading import MISTRAL3_CLASS

    real = AutoConfig.from_pretrained(str(REAL))
    text, vision = real.text_config.to_dict(), real.vision_config.to_dict()
    text.update(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16)
    vision.update(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, head_dim=16)
    config = Mistral3Config(text_config=text, vision_config=vision, image_token_index=real.image_token_index,
                            projector_hidden_act=real.projector_hidden_act,
                            spatial_merge_size=real.spatial_merge_size,
                            vision_feature_layer=real.vision_feature_layer,
                            multimodal_projector_bias=real.multimodal_projector_bias)
    torch.manual_seed(0)
    directory = tmp_path_factory.mktemp("mistral3") / "base"
    getattr(transformers, MISTRAL3_CLASS)(config).save_pretrained(directory)
    for name in ("tekken.json", "tokenizer_config.json", "special_tokens_map.json", "params.json"):
        if (REAL / name).is_file():
            shutil.copy2(REAL / name, directory / name)
    return directory


def _messages(text):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_job(tmp_path, base, *, stages):
    root = tmp_path / "job"
    shutil.copytree(base, root / "base")
    _write_jsonl(root / "data/train.jsonl", [{"prompt_messages": _messages(f"check {t}"), "chosen": ANSWER}
                                             for t in ("host", "node", "db", "printer")])
    _write_jsonl(root / "data/validation.jsonl", [{"prompt_messages": _messages("check nas"), "chosen": ANSWER}])
    _write_jsonl(root / "data/heldout.jsonl", [{"name": "router", "prompt_messages": _messages("check router"),
                                                "expected_tool": "ping", "expected_arguments": {},
                                                "max_new_tokens": 12}])
    payload = {"schema": "janus-training-job.v1", "job_id": "operations-20261119T000000Z", "role": "specialist",
               "model": {"source": "local", "path": "base"}, "dataset": {"source": "local", "path": "data"},
               "stages": stages, "evaluation": {"cases": "data/heldout.jsonl", "max_new_tokens": 12},
               "device": "cpu",
               "sft": {"epochs": 1, "learning_rate": 0.001, "lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
                       "max_length": 256, "batch_size": 2, "gradient_accumulation": 1, "evaluations_per_epoch": 1},
               "dpo": {"epochs": 1, "learning_rate": 0.0001, "max_length": 256, "gradient_accumulation": 1,
                       "pair_limit": 2},
               "guard": {"patience": 3, "min_improvement": 0.001}}
    parsed = TrainingJob.model_validate(payload)
    payload["admissions"] = [{"artifact": key, "receipt_sha256": "0" * 64} for key in parsed.input_artifacts()]
    (root / "job.json").write_text(json.dumps(payload), encoding="utf-8")
    return root / "job.json"


def test_prompts_are_the_exact_token_ids_serving_uses(tiny_mistral3):
    """Serving tokenizes the messages directly; text rendered and re-tokenized gives other ids."""
    from transformers import AutoTokenizer

    from learning.worker.prompts import encode_prompt

    tokenizer = AutoTokenizer.from_pretrained(str(tiny_mistral3))
    messages = _messages("check the router")
    served = list(tokenizer.apply_chat_template(messages, return_dict=True)["input_ids"])
    assert encode_prompt(tokenizer, messages) == served
    assert served[0] == tokenizer.bos_token_id                  # one BOS, from the template, not twice


def test_an_answer_ends_with_the_end_of_sequence_token(tiny_mistral3):
    from transformers import AutoTokenizer

    from learning.worker.data import encode_pair
    from learning.worker.prompts import encode_prompt

    tokenizer = AutoTokenizer.from_pretrained(str(tiny_mistral3))
    prompt = encode_prompt(tokenizer, _messages("check host"))
    feature = encode_pair(tokenizer, prompt, ANSWER, max_length=256)
    assert feature["input_ids"][: len(prompt)] == prompt
    assert feature["input_ids"][-1] == tokenizer.eos_token_id
    assert feature["labels"][: len(prompt)] == [-100] * len(prompt)
    assert feature["labels"][len(prompt):] == feature["input_ids"][len(prompt):]


def test_sft_and_dpo_train_the_language_tower_only(tmp_path, tiny_mistral3):
    result = run(make_job(tmp_path, tiny_mistral3, stages=["sft", "dpo"]), tmp_path / "out")
    assert result["outcome"] in {"trained", "no_improvement"}, result.get("error")
    sft = next(item for item in result["stages"] if item["stage"] == "sft")
    assert sft["ran"] is True and sft["total"] == 1
    adapter = tmp_path / "out" / "stages" / "sft" / "adapter" / "adapter_model.safetensors"
    from safetensors import safe_open

    with safe_open(str(adapter), "pt") as weights:
        names = list(weights.keys())
    assert names and all(".language_model." in name for name in names)   # never the vision tower
