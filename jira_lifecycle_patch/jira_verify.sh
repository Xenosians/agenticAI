#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/project/agenticaiPersonal/itsm-agent"
cd "$ROOT"

printf '\nChecking Jira milestone filenames...\n'
if find scripts services/jira services/ticketing tools/jira tools/ticketing tests/unit/services/jira tests/unit/tools/jira tests/unit/tools/ticketing tests/unit/subagents/llm/runtime \
  -type f \( -iname 'j2*' -o -iname 'j3*' -o -iname 'test_j2*' -o -iname 'test_j3*' \) -print | grep -q .; then
  echo 'ERROR: milestone-named Jira files remain.'
  find scripts services/jira services/ticketing tools/jira tools/ticketing tests/unit/services/jira tests/unit/tools/jira tests/unit/tools/ticketing tests/unit/subagents/llm/runtime \
    -type f \( -iname 'j2*' -o -iname 'j3*' -o -iname 'test_j2*' -o -iname 'test_j3*' \) -print
  exit 1
fi

printf '\nChecking Jira source for old milestone labels...\n'
if grep -RniE 'j2|j3' scripts/jira*.py services/jira services/ticketing/jira*.py tools/jira tests/unit/services/jira tests/unit/tools/jira tests/unit/tools/ticketing/test_jira*.py tests/unit/subagents/llm/runtime/test_jira_runtime_alignment.py --exclude='*.pyc'; then
  echo 'ERROR: old Jira milestone labels remain.'
  exit 1
fi

printf '\nCompiling Jira code...\n'
python -m compileall -q services/ticketing tools/ticketing services/jira tools/jira scripts tests/unit/tools/jira tests/unit/tools/ticketing tests/unit/services/jira tests/unit/subagents/llm/runtime

printf '\nRunning focused Jira lifecycle tests...\n'
pytest -q \
  tests/unit/tools/ticketing/test_jira_assignment_prepare.py \
  tests/unit/tools/ticketing/test_jira_assignment_execution.py \
  tests/unit/tools/ticketing/test_jira_comment_prepare.py \
  tests/unit/tools/ticketing/test_jira_comment_execution.py \
  tests/unit/tools/ticketing/test_jira_comment_mcp_execution.py \
  tests/unit/tools/ticketing/test_jira_transition_prepare.py \
  tests/unit/tools/ticketing/test_jira_transition_execution.py \
  tests/unit/tools/ticketing/test_jira_create_prepare.py \
  tests/unit/tools/ticketing/test_jira_create_execution.py \
  tests/unit/tools/ticketing/test_jira_create_mcp_execution.py \
  tests/unit/tools/ticketing/test_mutation_contract.py \
  tests/unit/tools/ticketing/test_mutations.py \
  tests/unit/tools/jira/test_jira_tools.py \
  tests/unit/services/jira \
  tests/unit/subagents/core/definitions/test_jira_specialist.py \
  tests/unit/subagents/core/definitions/test_ticket_jira_domain_partition.py \
  tests/unit/subagents/llm/runtime/test_jira_runtime_alignment.py

if [[ "${1:-}" == "--full" ]]; then
  printf '\nFocused Jira tests passed. Running full suite...\n'
  pytest -q
fi

printf '\nJira lifecycle verification passed.\n'
