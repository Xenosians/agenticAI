# Capability-Language Learning Architecture v2

## Purpose

Keep the existing agentic architecture intact while making the model better at
understanding the language users actually use for Jira, Git, and governed
developer/shell work.

This is intentionally **not** a hardcoded runtime router.

Runtime authority stays:

`user -> Hub LLM -> semantic intent -> specialist LLM -> SemanticGuard -> ToolGateway -> policy -> approval -> execution`

This patch changes the **learning surface**, not the execution authority.

## Core rule

The trusted capability catalogs and agent definitions are the source of truth.

Training examples are derived from:

- the exact tools currently owned by `jira-specialist`;
- the exact tools currently owned by `developer-specialist`;
- each tool's trusted description;
- its live argument schema and bounded enum values;
- trusted risk/effect/approval metadata;
- the existing generic process policy where applicable.

Adding a future capability to an agent definition automatically includes it in
the generated corpus. There is no separate runtime `if request contains X ->
call Y` table.

## Why authored language examples are not hardcoded routing

An LLM still requires supervision describing what language means. The
difference is where that mapping lives.

Bad runtime architecture:

```text
if "create jira" in request:
    call ticket_create
```

That bypasses the model and becomes an ever-growing manual parser.

This architecture:

```text
trusted capability metadata
        +
catalog-derived language variants
        +
typo/context/noise augmentation
        |
        v
     SFT / curriculum
        |
        v
model learns semantic mapping
        |
        v
SemanticGuard + ToolGateway remain authoritative
```

The training examples teach semantics. They do not execute or authorize them.

## Current generated coverage

### Jira specialist

Every capability currently declared in `subagents/agents/jira-specialist.md`
is included automatically, including ticket reads, ticket search/history,
comments, create/assign/transition, and project list/get/create/update/archive/
delete.

High-risk capabilities are intentionally part of the language curriculum. The
model must understand what "delete project" means; safety comes from trusted
policy + approval, not from pretending the capability does not exist.

### Developer specialist

Every capability currently declared in
`subagents/agents/developer-specialist.md` is included automatically:

- workspace reads/list/search/info;
- tests/build/process/service/log inspection;
- Git status/branches/log/diffs/changed files;
- staging/unstaging/stage-all;
- branch create/switch;
- commit/push;
- generic governed process execution.

The generic process examples are derived from the current
`services.process_runner.ALLOWED_EXECUTABLES` policy. The curriculum does not
create a second shell allowlist.

If a future destructive process such as remove/delete is added to trusted
process policy, give it an explicit policy shape and approval/risk metadata.
The curriculum can then include it from that trusted source. This patch
deliberately does not smuggle an ungoverned `rm` into `process_exec`.

## Language robustness

For every capability/argument shape the generator creates:

- canonical description-derived phrasing;
- conversational phrasing;
- terse capability-oriented phrasing;
- explicit contextual phrasing;
- noisy contextual phrasing;
- deterministic single-typo phrasing.

Typos are applied only to natural-language operation text. Exact project keys,
ticket keys, repository identities, paths, branch names, comments, summaries,
and other grounded values are never silently corrected. This preserves the
existing grounding contract.

## Hub + specialist learning

The generator emits two views.

### Worker SFT

Target:

```json
[
  {
    "name": "exact_capability_name",
    "arguments": {}
  }
]
```

This matches the current specialist output protocol.

### Hub curriculum lesson candidates

Target:

```json
{
  "delegations": [
    {
      "agent": "jira-specialist",
      "instructions": "...",
      "intent": {
        "summary": "...",
        "effect": "mutation",
        "allowed_tools": ["ticket_create"],
        "forbidden_tools": [],
        "allowed_arguments": {},
        "forbidden_arguments": {},
        "max_tool_calls": 1,
        "clarification_required": false
      }
    }
  ]
}
```

This matches the current Hub semantic contract.

Generated lesson files are intentionally only **candidates**. They still flow
through the existing curriculum review/materialization/training gates.

## Safety / destructive capabilities

Learning that a destructive operation exists is not authorization to execute
it.

For a mutation the runtime still has all existing boundaries:

1. Hub semantic intent.
2. Specialist exact capability call.
3. SemanticGuard verifies intent/tool/grounded-argument consistency.
4. ToolGateway checks trusted tool ownership, grounding, risk, policy, and
   approval.
5. Policy may bind immutable execution preconditions.
6. Approval stores an exact proposal.
7. Execution replays that exact proposal rather than asking the model again.

Therefore a high-risk `jira_project_delete` example can safely exist in
training data while still requiring the exact existing high-risk approval
path at runtime.

## Hardware strategy

The HP Victus should not be treated as a datacenter trainer.

Recommended sequence:

1. Build/audit the corpus. No model weights loaded.
2. Train Jira specialist with a bounded LoRA/QLoRA run first.
3. Evaluate on fixed Jira holdouts plus typo/context holdouts.
4. Only promote after the existing comparison gate passes.
5. Feed Hub lesson candidates through the existing reviewed curriculum path.
6. Train Hub in bounded phases.
7. Keep SWE-rebench for developer code-patching competence, not as the
   prerequisite for basic Jira/Git/shell semantic understanding.
8. Keep PPO/DPO/continual automation in the architecture, but do not make them
   prerequisites for this semantic bootstrapping step.

## Commands after patch installation

Build both language corpora:

```bash
python scripts/capability_curriculum_build.py --force
```

Audit the trusted guard metadata:

```bash
python scripts/capability_curriculum_audit.py
```

For a stricter audit where every current non-read capability must statically
require approval:

```bash
python scripts/capability_curriculum_audit.py --strict-mutation-approval
```

Rebuild the Jira SFT corpus. The existing Jira builder is patched to consume
catalog-derived language records instead of its historical manual seed table:

```bash
python scripts/jira_sft_corpus.py --force
python scripts/jira_sft_preflight.py
```

Run the existing bounded Jira training only when ready:

```bash
python scripts/jira_sft_train.py \
  --allow-training \
  --max-steps 80 \
  --gradient-accumulation-steps 8
```

For the Hub, inspect generated files under:

```text
.runtime/learning/capability-language/v2/<target>/hub-lessons/
```

They are compatible with the existing `run_curriculum_lessons` import shape
but still require review before dataset promotion.

Example:

```bash
python -m learning.cli.run_curriculum_lessons import \
  --curriculum hub-general-language-code-routing-v1 \
  --chapter hub-02-resource-operation \
  --file .runtime/learning/capability-language/v2/jira/hub-lessons/hub-02-resource-operation.jsonl
```

Review, promotion, membership, training, evaluation, and model activation
remain distinct.
