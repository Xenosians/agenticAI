# SRS 1.8 training and retrieval

Run commands from the AI repository with the `qwen-infra` Python environment.
`python scripts/training_methods.py` prints the supported method/role matrix.
Support means an implementation exists, not that every optimizer has been validated
on this laptop or that all SRS acceptance criteria are complete.

| Method | Entry point | Admitted input | Current execution evidence |
|---|---|---|---|
| SFT + PEFT/LoRA/QLoRA | `scripts/train_recipe.py --recipe recipe.json --allow-training` | Reviewed, hash-pinned messages; independent train/validation | Networking uses the existing `specialist_sft_train.py` QLoRA trainer on the RTX 4050 |
| Self-supervised causal adaptation | Same recipe CLI, objective `causal_adaptation` | Reviewed licensed text; next-token loss | Implemented; optimizer run not performed |
| Full parameter tuning | Same recipe CLI, tuning `full` | Same objective-specific evidence | Implemented; conservatively requires at least 16 GiB free VRAM |
| DPO | `scripts/train_preferences.py --settings settings.json --train TRAIN --validation VALIDATION --allow-training` | Existing verified preference partitions with matching base/tokenizer provenance | Real train/evaluate/save path; not run on this laptop |
| PPO | `scripts/train_ppo.py --checkpoint-id ID --cycle-id ID --training-suite router-training-v1 --settings settings.json --allow-training` | Verified Hub adapter and separate training episodes | Existing bounded sequence PPO, deterministic routing reward and value head; no live tool execution |
| RAG | `KNOWLEDGE_BACKEND=local` | Reviewed public JSONL snapshot and SHA256 | Actual lexical FTS5 retrieval; not weight training |
| Continual learning | `python -m learning.cli.run_continual_cycle --help` and existing automation service | Reviewed evidence, replay, checkpoint and promotion contracts | Governed orchestration; no automatic production promotion added |

PEFT is a family of parameter-efficient techniques; LoRA and QLoRA are tuning
configurations, not additional learning objectives. LangChain is not a training method.
DPO and PPO default to a 12 GiB free-VRAM preflight in their new CLI entry points.
These are conservative policies, not claims that every model needs that much memory.

## Generic recipe

The recipe schema is `learning.training.recipes.AdaptationRecipe`. Required fields:
`objective`, `model_path`, `dataset`, `dataset_sha256`, `allowed_sources`,
`allowed_licenses`, `output_root`. `tuning` defaults to `qlora`; `role` defaults
to `specialist`. Paths must refer to acquired local weights and separate output.
Run without `--allow-training` to validate the dataset without loading a model.

Each JSONL row requires `id`, `source`, `license`, `reviewed: true`, `split`
(`train` or `validation`), plus `messages` for SFT or `text` for causal adaptation.
Duplicate content, cross-split leakage, changed hashes, malformed records and
privacy-scanner findings are rejected. SFT masks the prompt loss; oversized
examples fail instead of silently losing their assistant target. Outputs include
metrics and artifact hashes and remain candidates.

## Public retrieval snapshot

Set `KNOWLEDGE_SNAPSHOT_PATH` and `KNOWLEDGE_SNAPSHOT_SHA256` alongside the local
backend. Each JSONL document requires `document_id`, `kind` (`knowledge` or
`runbook`), `title`, `summary`, `content`, `source`, `license`, `reviewed: true`,
and `visibility: public`; `tags` is optional. Search returns observed documents
and provenance. Private documents fail admission: the current service contract
has no authenticated caller identity. This is not full private enterprise RAG.
Documents are untrusted content and cannot authorize tools.

## Networking evaluation

Base model: `Qwen/Qwen2.5-0.5B-Instruct`, pinned revision
`7ae557604adf67be50417f59c2c2f167def9a775`, Apache-2.0. Base weights and adapters
live under `../models/`, outside the AI repository. Acquisition provenance is
stored beside the model. The original guided test improved from 1/8 to 8/8,
but an independent unhinted check revealed a training-context shortcut. Its
original report is retained in `.runtime/evaluation/networking-unhinted-v1.json`.
The revised training adds unhinted positive examples from the training partition
only; held-out prompts and expected outputs are unchanged. Both evaluations and
human review of negative responses are required before production promotion.
