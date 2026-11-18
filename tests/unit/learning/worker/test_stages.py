"""Every training stage really runs on a tiny model: merge, SFT under the fit guard, DPO and PPO.

The held-out gate decides what is kept; nothing here mocks the training libraries.
"""
import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("trl")

from learning.worker.job import TrainingJob  # noqa: E402
from learning.worker.loading import NotAdmitted, admitted_load  # noqa: E402
from learning.worker.stages import run  # noqa: E402

ANSWER = '[{"arguments":{},"name":"ping"}]'
SYSTEM = "Answer with one tool call."


def _tiny_base(directory: Path, initializer_range: float = 0.2) -> Path:
    """A 2-layer Qwen2 with a byte-level tokenizer.

    Its tied embedding is initialised wider than the 0.02 default: LoRA leaves the embedding and the final
    norm frozen, and at 0.02 the largest reachable logit is about 1.3, so no adapter could ever fit the
    answer. A trained model's embedding has the range this gives."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {character: index for index, character in enumerate(alphabet)}
    vocab["<pad>"], vocab["</s>"] = len(vocab), len(vocab) + 1
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="</s>", pad_token="<pad>")
    config = Qwen2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                         tie_word_embeddings=True, eos_token_id=fast.eos_token_id, pad_token_id=fast.pad_token_id,
                         bos_token_id=None, initializer_range=initializer_range)
    torch.manual_seed(0)
    Qwen2ForCausalLM(config).save_pretrained(directory)
    fast.save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    return _tiny_base(tmp_path_factory.mktemp("tiny") / "base")


def _messages(text):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


# Requests of different lengths, so the answer cannot be learned by its position in the prompt.
TRAIN_TARGETS = ("host", "database server", "node", "mail gateway", "db", "file share", "printer", "vpn")
VALIDATION_TARGETS = ("web server", "nas", "backup box")
HELDOUT_TARGETS = ("router", "core switch")


def make_job(tmp_path, tiny_base, *, stages, role="specialist", admit=True, **overrides):
    root = tmp_path / "job"
    shutil.copytree(tiny_base, root / "base")
    _write_jsonl(root / "data/train.jsonl",
                 [{"prompt_messages": _messages(f"check {target} {n}"), "chosen": ANSWER}
                  for n in range(2) for target in TRAIN_TARGETS])
    _write_jsonl(root / "data/validation.jsonl",
                 [{"prompt_messages": _messages(f"check {target}"), "chosen": ANSWER} for target in VALIDATION_TARGETS])
    _write_jsonl(root / "data/heldout.jsonl", [
        {"name": target, "prompt_messages": _messages(f"check {target}"), "expected_tool": "ping",
         "expected_arguments": {}, "max_new_tokens": 40} for target in HELDOUT_TARGETS])
    payload = {
        "schema": "janus-training-job.v1", "job_id": "tiny-20261119T000000Z", "role": role,
        "model": {"source": "local", "path": "base"},
        "dataset": {"source": "local", "path": "data"},
        "stages": stages, "evaluation": {"cases": "data/heldout.jsonl", "max_new_tokens": 40},
        "device": "cpu",
        "sft": {"epochs": 40, "learning_rate": 0.01, "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.0,
                "max_length": 256, "batch_size": 4, "gradient_accumulation": 1, "evaluations_per_epoch": 1},
        "dpo": {"epochs": 2, "learning_rate": 0.001, "max_length": 256, "gradient_accumulation": 1,
                "pair_limit": 4},
        "ppo": {"epochs": 1, "learning_rate": 0.0001, "samples_per_prompt": 2, "max_new_tokens": 16},
        "guard": {"patience": 3, "min_improvement": 0.001},
    }
    payload.update(overrides)
    if admit:
        parsed = TrainingJob.model_validate(payload)
        payload["admissions"] = [{"artifact": key, "receipt_sha256": "0" * 64} for key in parsed.input_artifacts()]
    (root / "job.json").write_text(json.dumps(payload), encoding="utf-8")
    return root / "job.json"


def stage(result, name):
    return next(item for item in result["stages"] if item["stage"] == name)


def test_a_job_without_the_hosts_admissions_never_loads_anything(tmp_path, tiny_base):
    job = make_job(tmp_path, tiny_base, stages=["sft"], admit=False)
    result = run(job, tmp_path / "out")
    assert result["outcome"] == "refused"
    assert "did not admit" in result["error"]
    assert not (tmp_path / "out/stages").exists()


def test_admitted_load_refuses_an_artifact_the_job_does_not_list(tmp_path, tiny_base):
    from learning.worker.job import load_job

    parsed, _ = load_job(make_job(tmp_path, tiny_base, stages=["sft"]))
    with pytest.raises(NotAdmitted):
        admitted_load(parsed, "huggingface:someone/else@main", lambda: pytest.fail("loaded"))


def test_sft_learns_the_task_under_the_fit_guard_and_is_kept_because_it_beats_the_base(tmp_path, tiny_base):
    result = run(make_job(tmp_path, tiny_base, stages=["sft"]), tmp_path / "out")
    sft = stage(result, "sft")
    assert result["baseline"]["passed"] == 0
    assert sft["ran"] and sft["kept"], json.loads((tmp_path / "out/evaluations/sft.json").read_text())["cases"]
    assert sft["passed"] == sft["total"] == 2
    assert sft["guard"]["evaluations"] >= 2
    assert sft["guard"]["best_validation_loss"] < sft["guard"]["first_validation_loss"]
    assert result["outcome"] == "trained"
    final = tmp_path / "out" / result["final_adapter"]
    assert (final / "adapter_config.json").is_file() and (final / "adapter_model.safetensors").is_file()
    assert len(result["final_adapter_sha256"]) == 64
    assert not list((tmp_path / "out/stages/sft").glob("checkpoint*"))   # checkpoints are cleaned up


def test_a_stage_that_does_not_beat_the_previous_one_is_dropped_and_the_earlier_model_stays(tmp_path, tiny_base):
    job = make_job(tmp_path, tiny_base, stages=["sft", "dpo"])
    _write_jsonl(job.parent / "data/preferences.jsonl", [
        {"prompt_messages": _messages(f"check {target}"), "chosen": ANSWER, "rejected": "I cannot help."}
        for target in TRAIN_TARGETS[:4]])
    result = run(job, tmp_path / "out")
    sft, dpo = stage(result, "sft"), stage(result, "dpo")
    assert sft["kept"]
    assert dpo["ran"] and dpo["pairs"] >= 1
    assert dpo["kept"] is False and dpo["reason"] == "not_better"
    assert result["final_stage"] == "sft"
    from learning.worker.stages import fingerprint_directory

    assert result["final_adapter_sha256"] == fingerprint_directory(tmp_path / "out" / sft["adapter"])


def test_dpo_trains_on_the_jobs_preference_pairs(tmp_path, tiny_base):
    job = make_job(tmp_path, tiny_base, stages=["dpo"], dpo={"epochs": 1, "learning_rate": 0.001, "max_length": 256,
                                                               "gradient_accumulation": 1,
                                                               "self_generated_pairs": False})
    _write_jsonl(job.parent / "data/preferences.jsonl", [
        {"prompt_messages": _messages(f"check {target}"), "chosen": ANSWER, "rejected": "I cannot help."}
        for target in TRAIN_TARGETS[:4]])
    result = run(job, tmp_path / "out")
    dpo = stage(result, "dpo")
    assert dpo["ran"] and dpo["pairs"] == 4
    assert dpo["kept"] in (True, False) and dpo["reason"] in {"better", "not_better", "nothing_passed"}
    adapter = tmp_path / "out" / dpo["adapter"]
    assert (adapter / "adapter_model.safetensors").is_file()
    assert not (adapter / "ref").exists()      # the frozen reference copy is never saved as the result


def test_dpo_without_any_pairs_is_skipped_not_failed(tmp_path, tiny_base):
    job = make_job(tmp_path, tiny_base, stages=["dpo"], dpo={"self_generated_pairs": False})
    result = run(job, tmp_path / "out")
    assert stage(result, "dpo")["ran"] is False and "no preference pairs" in stage(result, "dpo")["reason"]
    assert result["outcome"] == "no_improvement"


def test_dpo_mines_the_models_own_wrong_answers_as_rejected_examples(tmp_path, tiny_base):
    result = run(make_job(tmp_path, tiny_base, stages=["dpo"]), tmp_path / "out")
    dpo = stage(result, "dpo")
    assert dpo["ran"] and dpo["self_generated"] and dpo["pairs"] >= 1, dpo


def test_the_weight_space_merge_combines_adapters_of_the_same_base(tmp_path, tiny_base):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    job = make_job(tmp_path, tiny_base, stages=["merge"], merge={"method": "ties_svd", "density": 0.5})
    for name, rank in (("a", 4), ("b", 8)):
        model = AutoModelForCausalLM.from_pretrained(tiny_base)
        peft = get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank * 2, target_modules="all-linear",
                                                init_lora_weights=False, task_type="CAUSAL_LM"))
        peft.save_pretrained(job.parent / "adapters" / name)
    result = run(job, tmp_path / "out")
    merge = stage(result, "merge")
    assert merge["ran"] and merge["adapters"] == 2 and merge["method"] == "ties_svd"
    config = json.loads((tmp_path / "out" / merge["adapter"] / "adapter_config.json").read_text())
    assert config["r"] == 8
    from safetensors.torch import load_file

    weights = load_file(str(tmp_path / "out" / merge["adapter"] / "adapter_model.safetensors"))
    assert weights and any(float(tensor.abs().sum()) > 0 for tensor in weights.values())


def test_ppo_optimises_the_tool_call_reward_for_the_hub(tmp_path, tiny_base):
    result = run(make_job(tmp_path, tiny_base, stages=["sft", "ppo"], role="hub"), tmp_path / "out")
    ppo = stage(result, "ppo")
    assert ppo["ran"], ppo
    assert ppo["method"].startswith("ppo-clip")
    assert ppo["steps"] >= 1 and "mean_reward" in ppo
    assert ppo["reason"] in {"better", "not_better", "nothing_passed"}


def test_a_stop_file_interrupts_the_run_and_keeps_no_half_trained_model(tmp_path, tiny_base):
    stop = tmp_path / "stop"
    stop.write_text("yield")
    result = run(make_job(tmp_path, tiny_base, stages=["sft"]), tmp_path / "out", stop_file=stop)
    assert result["outcome"] == "interrupted"
    assert result.get("final_adapter") is None


def test_the_result_names_the_machine_it_ran_on(tmp_path, tiny_base):
    result = run(make_job(tmp_path, tiny_base, stages=["dpo"], dpo={"self_generated_pairs": False}),
                 tmp_path / "out")
    worker = result["worker"]
    assert worker["device"] == "cpu" and worker["cpu_count"] >= 1
    assert {"torch", "transformers", "peft", "trl"} <= set(worker["versions"])
    assert json.loads((tmp_path / "out/result.json").read_text())["job_id"] == "tiny-20261119T000000Z"
