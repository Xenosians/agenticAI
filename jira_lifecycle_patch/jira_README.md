# Jira Lifecycle Patch

This bundle replaces milestone-style Jira implementation names with functional Jira names and installs the complete governed Jira issue-mutation lifecycle.

Run from the repository root:

```bash
cd /mnt/c/project/agenticaiPersonal/itsm-agent
python /mnt/c/project/agenticaiPersonal/itsm-agent/jira_lifecycle_patch/jira_install.py
bash /mnt/c/project/agenticaiPersonal/itsm-agent/jira_lifecycle_patch/jira_verify.sh
```

Use `jira_verify.sh --full` to run the complete repository test suite after the focused Jira verification.

The installer removes old extracted Jira milestone bundles, old milestone proof databases, Python caches, and phase-coded Jira scripts/tests. Useful proof/evaluation behavior is preserved under functional `jira_*` filenames.
