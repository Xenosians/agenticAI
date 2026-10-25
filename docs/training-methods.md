# Training methods and retrieval

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
| PPO | `scripts/train_ppo.py --checkpoint-id ID --cycle-id ID --training-suite router-routing --settings settings.json --allow-training` | Verified Hub adapter and separate training episodes | Existing bounded sequence PPO, deterministic routing reward and value head; no live tool execution |
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

## Native templates and adapter scope

The shared Hub SFT/DPO/PPO encoder also rejects oversized examples and requires
the full serving prompt to remain a prefix of the completed training sequence.
It masks every prompt token and retains the complete assistant answer and EOS.
Native template errors fail the run instead of silently changing serialization.
Transformers mappings and the Mistral native assistant-continuation API are
supported; a generic text fallback is reserved for tokenizers without a native
chat template.

Text adaptation of Mistral3 checkpoints targets language projections. The vision
tower and multimodal projector remain frozen. Fresh and resumed adapters check
their actual trainable parameter scope before optimization; unsafe existing
adapters are rejected without rewriting their saved targets or ranks.

Training requirements pin `datasets==5.0.1`, matching the existing DPO API gate.
Other dependency declarations are not a complete reproducible environment lock.
These compatibility and rejection checks do not establish model quality or mean
that a new SFT, DPO or PPO job has run.

The Ministral specialist's suffix-only loss sums supervised-token loss and
normalizes it by the entire gradient-accumulation window's target count. Unequal
target lengths and a short final window therefore retain the same weighting as
full causal loss. CPU Trainer/LoRA gradient tests verify this calculation; they
do not establish GPU performance or distributed training acceptance.

## Jira corpus preparation

`jira_sft_corpus.py` preserves the historical `legacy` layout by default. To
prepare a separate candidate using the runtime's isolated local task layout:

```sh
python scripts/jira_sft_corpus.py --output /path/to/new-corpus \
  --worker-prompt-profile compact_grounded --prompt-layout runtime_task
python scripts/jira_sft_corpus.py --output /path/to/new-corpus \
  --worker-prompt-profile compact_grounded --verify-only
```

This layout narrows each capability catalog to its declared tool, binds only
grounded argument fields and places the original request last. Verification
reconstructs each record's messages from current trusted metadata and checks
split hashes. Training manifests retain the layout; historical artifacts remain
readable. It covers isolated local tasks, without prior conversation, outbound
connection role changes or verified account continuation content.

Jira encoding now uses the same strict prompt/completion boundary as Hub
training. Mandatory instructions and target tokens are never clipped. Native
templates own their EOS; the fallback adds one terminal EOS. Oversized examples
identify the record and fail admission. A corpus schema check alone does not
establish that its sequences fit the chosen training budget.

With the cached Jira tokenizer, the 376-record candidate spans 1,172–1,402
tokens; 23 exceed the CLI's 1,280-token default. The old full-catalog candidate
spans 1,773–1,963 tokens, all above that limit. Neither corpus fits the trainer
function's 1,024-token default. That tokenizer-only admission check discarded no
records and performed no optimization. A training run must declare its sequence
budget and resource settings separately; this measurement is not permission to
increase either.

The existing merged model passed 24/33 isolated protocol cases with its legacy
prompt and 3/33 with the experimental grounded prompt. Both quality gates
failed. Preparing a corrected corpus does not activate that prompt, retrain the
model, prove live Jira execution or complete the MVP.

## Jira training guard

`jira_sft_train.py` measures validation loss before training and every
`--checkpoint-interval` optimizer steps, saves a checkpoint at the same
interval and keeps the newest and the best. It stops after
`--early-stopping-patience` checkpoints without a lower validation loss and
ends on the best adapter. `--resume-run RUN_DIR` continues an interrupted run
from its newest checkpoint only when the run's `run-config.json` (corpus
hashes, prompt layout, step budget, sequence length, LoRA and optimizer
settings) matches. The training manifest records the validation summary.
The shared specialist trainer (`specialist_sft.py`) still has no validation
checkpoint selection.

`jira_sft_preflight.py` accepts the same candidate prompt and numeric recipe
options as `jira_sft_train.py` (the defaults are unchanged). Pass the same
`--worker-prompt-profile`, `--max-length`, `--max-steps`, learning rate,
accumulation, LoRA and checkpoint/patience options to both commands. Preflight
verifies both corpus splits and uses the trainer's strict encoder, including
saved `runtime_task` messages, complete targets and template/EOS boundaries.
An oversized record rejects the check by identity rather than being clipped or
reported as ready. It also requires CUDA/BF16 and instantiates the actual
validation/checkpoint `TrainingArguments`; a pass establishes prerequisites,
not optimizer completion, a quality gate or live tool execution. Tokenizer-only
sequence admission remains separate from hardware readiness. No preflight
operation changes the serving prompt or model binding.

New Jira runs also pin the resolved base-model path and fresh content hashes for
the checkpoint, configuration and tokenizer. Resume checks bypass the runtime's
fingerprint cache, so replacing bytes at the same path is a mismatch. Older runs
without that recorded identity cannot establish which original base was used;
they are preserved for reconciliation and cannot be made resumable by assigning
today's fingerprint to their old configuration.

Recipe validation and resume-file checks precede CUDA checks and model loading.
The newest checkpoint must retain its adapter, optimizer, scheduler, RNG and
Trainer state, plus the validation-best adapter and matching early-stop state.
An incomplete newest checkpoint is rejected without falling back to an older
one. These are metadata and nonempty-file checks; Trainer still validates and
loads the binary states later. Completed optimization or an existing training
manifest requires artifact reconciliation instead of repeating optimizer steps.
The initial validation loss is diagnostic: the untrained adapter is not a saved
best-checkpoint candidate, and a low validation loss does not replace held-out
tool evaluation or authorized live acceptance.

## Jira evaluation case sets

`jira_sft_register_candidate.py` registers a trained checkpoint as the separate
`jira-func-candidate` profile. It refuses any profile key an agent definition
serves, such as the specialist's `jira-func-trained`. Changing production is
then a separate promotion step, taken only after both evaluations.

`jira_specialist_model_eval.py` scores the built-in `development` cases by
default. They guided corpus changes, so a pass there is development evidence.
`--cases learning/evaluation/evals/jira-specialist-holdout.json` scores the
held-out set instead. Its coverage plan was fixed before any candidate was
scored on it. A unit test rejects any held-out request that shares a request
template or a free-text value (ticket key, label, search text, name) with a
training record.
Shared values from closed vocabularies are allowed (status, priority, ticket
type, template, relationship). Every report records `case_set` and
`case_set_sha256`. Promotion needs both sets at a strict pass. Do not tune
the corpus on held-out failures: that turns the set into development data, and
promotion then needs a newly written held-out set.

### Account-to-ticket continuation

After an approved account creation, the Jira specialist receives one extra
system message: the verified `user_id` and email, with the instruction to
compose the ticket summary from them. `ticket_continuation_message` in
`agent/onboarding.py` builds it for the runtime, the corpus and the evaluation,
so all three render the same prompt. The runtime-task corpus includes
continuation records (`verified_account` set) whose target summary names that
account. The legacy layout cannot carry this message and omits them.

`--cases learning/evaluation/evals/jira-continuation-holdout.json` scores this
step. A proposal passes only with the bound project copied exactly, a
non-empty summary naming the verified `user_id` or email, and no other
argument. A unit test rejects any of these cases whose request, account or
project appears in training. Continuation prompts reach 1,432 tokens, so train
with `--max-length 1536`; serving does not truncate the prompt.

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
