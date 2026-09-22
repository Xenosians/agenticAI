# Agentic Git checkpoint → push → switch bundle

This is a **safe overlay installer**, not a blind replacement of your whole repo. That is intentional: your local `itsm-agent` already contains uncommitted work that is newer than GitHub, so replacing entire existing files from a ZIP could destroy those changes.

The installer adds the new workflow files and patches only the required integration points in-place. Before touching existing files, it creates a timestamped backup folder in the repo.

## Assumed prerequisites

The bundle expects the two checkpoints you already made green:

- `workspace_git_stage_all`
- `workspace_git_commit`

It also expects the existing clean-tree `workspace_git_switch_branch` primitive.

## What it installs

The user-facing switch remains the same semantic capability:

```text
workspace_git_switch_branch(repository, branch_name)
```

Trusted policy chooses the execution path.

### Clean working tree

```text
approve
  → git switch --no-guess <target>
  → verify branch + HEAD + clean tree
```

### Dirty working tree

```text
one HIGH-risk approval
  → verify exact approval-bound worktree snapshot
  → git add -A
  → deterministic checkpoint commit
  → push exact new commit to configured upstream
  → verify remote SHA == new commit
  → git switch --no-guess <target>
  → verify final branch
```

If add, commit, push, or push verification fails, **the branch switch is not attempted**.

The deterministic checkpoint message is:

```text
checkpoint before switching to <target-branch>
```

## Push boundary

The new standalone `workspace_git_push` and the dirty switch workflow enforce:

- current attached branch only
- that branch's already-configured upstream only
- HTTPS or SSH configured push URL only
- no force push
- no tags
- no model-selected remote
- no model-selected refspec
- pre-push hooks disabled
- approval binds branch, HEAD, remote, remote URL, upstream ref, and remote HEAD
- execution fails closed if any bound state changes before approval is executed

For automatic dirty checkpoint-and-switch, v1 additionally requires:

```text
local current HEAD == configured remote branch HEAD
```

before checkpointing. This prevents the workflow from silently pushing older unrelated local commits together with the generated checkpoint.

## Files added

- `tools/git/workflow.py`
- `tools/git/workflow_catalog.py`
- `tests/unit/tools/git/test_workflow_push_switch.py`
- `tests/unit/subagents/core/tooling/test_policy_execution_binding.py`
- `agentic_git_bundle_verify.sh`

## Existing files patched

- `subagents/core/tooling/gateway.py`
- `subagents/core/orchestration/runtime.py`
- `tools/git/catalog.py`
- `tools/git/mcp.py`
- `subagents/agents/developer-specialist.md`

## Install

Extract this ZIP anywhere. Then:

```bash
cd /mnt/c/project/agenticaiPersonal/itsm-agent
conda activate qwen-infra

python /path/to/agentic_git_checkpoint_switch_bundle/install.py
bash agentic_git_bundle_verify.sh
```

The installer creates a backup such as:

```text
.agentic_git_bundle_backup_20260922-XXXXXX/
```

before patching existing files.

Do **not** push anything just because the bundle installed successfully. First run the verification suite and then do a controlled live smoke.
