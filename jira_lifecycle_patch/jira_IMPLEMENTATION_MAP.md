# Jira Cleanup / Naming Map

Retired milestone filenames are consolidated as follows:

- archive corrected proposal -> `scripts/jira_project_archive_approval_proof.py` (replaces the older duplicate archive proof)
- archive fixture create -> `scripts/jira_project_archive_fixture_create.py`
- delete approval proof -> `scripts/jira_project_delete_approval_proof.py`
- delete fixture create -> `scripts/jira_project_delete_fixture_create.py`
- ticket read E2E -> `scripts/jira_ticket_read_e2e.py`
- ticket create proposal -> `scripts/jira_ticket_create_proposal.py`
- ticket create execution -> `scripts/jira_ticket_create_execute.py`
- Jira specialist model evaluation -> `scripts/jira_specialist_model_eval.py`
- Jira project tool tests -> `tests/unit/tools/jira/test_jira_tools.py`
- Jira runtime alignment tests -> `tests/unit/subagents/llm/runtime/test_jira_runtime_alignment.py`

Generated Jira mutation policy modules remain functional names:

- `jira_issue_policy.py`
- `jira_comment_policy.py`
- `jira_assignment_policy.py`
- `jira_transition_policy.py`
- existing `jira_create_policy.py`

Disposable milestone approval SQLite databases and extracted prior patch bundles are removed rather than renamed because they are generated proof/install state, not source-of-truth code.
