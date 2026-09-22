#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-/mnt/c/project/agenticaiPersonal/itsm-agent}"
cd "$REPO"

echo "== Python syntax =="
python -m py_compile \
  services/atlassian/__init__.py \
  services/atlassian/core.py \
  tools/atlassian/__init__.py \
  tools/atlassian/catalog.py \
  tools/atlassian/mcp.py \
  scripts/atlassian_preflight.py

echo
echo "== J1 targeted tests =="
pytest -q \
  tests/unit/services/atlassian \
  tests/unit/tools/atlassian \
  tests/unit/config/test_atlassian_settings.py

echo
echo "== Existing capability regression =="
if [[ -f tests/unit/subagents/core/tooling/test_capability_driven_agents.py ]]; then
  pytest -q tests/unit/subagents/core/tooling/test_capability_driven_agents.py
fi

echo
echo "== Whitespace =="
git diff --check

echo
echo "J1 Atlassian bundle verification passed."
