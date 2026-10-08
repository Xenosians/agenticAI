#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/project/agenticaiPersonal/itsm-agent

python scripts/v18_03_acceptance.py \
  --mode repo
