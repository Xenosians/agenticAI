from pathlib import Path

import pytest

from learning.evaluation.developer_behavioral_holdout import (
    DeveloperBehavioralHoldoutRecord,
)

from learning.evaluation.developer_behavioral_sandbox import (
    _sandbox_script,
    _test_commands,
)


def _record() -> DeveloperBehavioralHoldoutRecord:

    return (
        DeveloperBehavioralHoldoutRecord(
            record_id="behavior-case",

            parent_record_id="parent-case",

            source_id="source",

            source_record_id="owner__repo-1",

            source_identity=(
                "instance:owner__repo-1"
            ),

            upstream_index=1,

            problem_statement=(
                "Fix the bug."
            ),

            repository="owner/repo",

            base_commit=(
                "0123456789abcdef"
                "0123456789abcdef"
                "01234567"
            ),

            repository_license="MIT",

            language="Python",

            interface=None,

            image_name=(
                "example/image:tag"
            ),

            test_patch=(
                "diff --git a/test.py b/test.py\n"
                "+# test"
            ),

            fail_to_pass=[
                "test_bug"
            ],

            pass_to_pass=[
                "test_existing"
            ],

            install_config={
                "test_cmd": [
                    "pytest -q"
                ],

                "log_parser":
                    "parse_log_pytest",
            },

            reference_patch_sha256=(
                "0"
                * 64
            ),

            gold_patch_included=False,

            evaluation_only=True,

            training_eligible=False,
        )
    )


def test_test_commands_come_from_install_config():

    assert (
        _test_commands(
            _record()
        )
        == [
            "pytest -q"
        ]
    )


def test_sandbox_script_applies_candidate_before_hidden_test_patch():

    script = (
        _sandbox_script(
            record=(
                _record()
            ),

            commands=[
                "pytest -q"
            ],

            nonce="unit",
        )
    )

    candidate = (
        script.index(
            "/patches/candidate.diff"
        )
    )

    hidden_test = (
        script.index(
            "/patches/test.diff"
        )
    )

    tests = (
        script.index(
            "pytest -q"
        )
    )

    assert (
        candidate
        < hidden_test
        < tests
    )


def test_sandbox_script_resets_exact_base_commit():

    record = (
        _record()
    )

    script = (
        _sandbox_script(
            record=record,
            commands=[
                "pytest -q"
            ],
            nonce="unit",
        )
    )

    assert (
        (
            "git reset --hard "
            + record.base_commit
        )
        in script
    )


def test_missing_test_command_fails_closed():

    record = (
        _record()
    )

    record.install_config = {
        "log_parser":
            "parse_log_pytest"
    }

    with pytest.raises(
        ValueError,
        match="test_cmd",
    ):

        _test_commands(
            record
        )
