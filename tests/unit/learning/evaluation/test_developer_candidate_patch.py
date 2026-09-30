import json

from learning.evaluation.developer_behavioral_holdout import (
    DeveloperBehavioralHoldoutRecord,
)

from learning.evaluation.developer_behavioral_sandbox import (
    DockerImageIdentity,
)

from learning.evaluation.developer_candidate_patch import (
    CandidateContextFile,
    DeveloperCandidateContext,
    _build_messages,
    _candidate_visible_case,
    _context_excerpt,
    _context_path_allowed,
    _extract_candidate_patch,
    _issue_keywords,
    _issue_referenced_paths,
    _rank_context_paths,
)


def _record() -> DeveloperBehavioralHoldoutRecord:

    return (
        DeveloperBehavioralHoldoutRecord(
            record_id="record",

            parent_record_id="parent",

            source_id="source",

            source_record_id=(
                "owner__repo-1"
            ),

            source_identity=(
                "instance:owner__repo-1"
            ),

            upstream_index=1,

            problem_statement=(
                "Fix WidgetParser when empty "
                "values are provided."
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
                "SECRET-HIDDEN-TEST-PATCH"
            ),

            fail_to_pass=[
                "SECRET-FAIL-TO-PASS"
            ],

            pass_to_pass=[
                "SECRET-PASS-TO-PASS"
            ],

            install_config={
                "test_cmd": [
                    "SECRET-TEST-COMMAND"
                ],

                "log_parser":
                    "parse_log_pytest",
            },

            reference_patch_sha256=(
                "a"
                * 64
            ),

            gold_patch_included=False,

            evaluation_only=True,

            training_eligible=False,
        )
    )


def _context() -> DeveloperCandidateContext:

    record = (
        _record()
    )

    excerpt = (
        "class WidgetParser:\n"
        "    pass\n"
    )

    return (
        DeveloperCandidateContext(
            behavioral_holdout_id=(
                "behavioral"
            ),

            behavioral_records_sha256=(
                "a"
                * 64
            ),

            parent_holdout_id=(
                "loss-holdout"
            ),

            parent_records_sha256=(
                "b"
                * 64
            ),

            source_id="source",

            revision="revision",

            record_id=(
                record.record_id
            ),

            source_record_id=(
                record.source_record_id
            ),

            source_identity=(
                record.source_identity
            ),

            problem_statement=(
                record.problem_statement
            ),

            repository=(
                record.repository
            ),

            base_commit=(
                record.base_commit
            ),

            language=(
                record.language
            ),

            image=(
                DockerImageIdentity(
                    requested=(
                        record.image_name
                    ),

                    image_id=(
                        "sha256:"
                        + (
                            "c"
                            * 64
                        )
                    ),

                    repo_digests=[],
                )
            ),

            repository_tree=[
                "src/widget.py",
                "tests/test_widget.py",
            ],

            files=[
                CandidateContextFile(
                    path=(
                        "src/widget.py"
                    ),

                    excerpt=(
                        excerpt
                    ),

                    captured_chars=len(
                        excerpt
                    ),

                    excerpt_chars=len(
                        excerpt
                    ),

                    truncated=False,
                )
            ],

            base_commit_verified=True,

            hidden_evaluation_metadata_included=False,

            evaluation_only=True,
        )
    )


def test_candidate_visible_case_excludes_hidden_evaluator_metadata():

    visible = (
        _candidate_visible_case(
            _record()
        )
    )

    serialized = (
        json.dumps(
            visible
        )
    )

    assert (
        "SECRET-HIDDEN-TEST-PATCH"
        not in serialized
    )

    assert (
        "SECRET-FAIL-TO-PASS"
        not in serialized
    )

    assert (
        "SECRET-PASS-TO-PASS"
        not in serialized
    )

    assert (
        "SECRET-TEST-COMMAND"
        not in serialized
    )


def test_model_messages_contain_only_candidate_context():

    messages = (
        _build_messages(
            _context()
        )
    )

    serialized = (
        json.dumps(
            messages
        )
    )

    assert (
        "Fix WidgetParser"
        in serialized
    )

    assert (
        "src/widget.py"
        in serialized
    )

    for secret in [
        "SECRET-HIDDEN-TEST-PATCH",
        "SECRET-FAIL-TO-PASS",
        "SECRET-PASS-TO-PASS",
        "SECRET-TEST-COMMAND",
    ]:

        assert (
            secret
            not in serialized
        )


def test_model_messages_preserve_training_task_framing():

    messages = (
        _build_messages(
            _context()
        )
    )

    assert (
        messages[
            0
        ][
            "content"
        ]
        .startswith(
            "You are a software-engineering agent."
        )
    )

    assert (
        messages[
            1
        ][
            "content"
        ]
        .startswith(
            "Resolve the software issue below "
            "with a minimal, testable source patch."
        )
    )


def test_extract_candidate_patch_from_fence():

    raw = """
Here is the patch:

```diff
diff --git a/a.py b/a.py
index 1111111..2222222 100644
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-old
+new
```
"""

    patch = (
        _extract_candidate_patch(
            raw
        )
    )

    assert (
        patch.startswith(
            "diff --git a/a.py b/a.py"
        )
    )

    assert (
        patch.endswith(
            "\n"
        )
    )

    assert (
        "```"
        not in patch
    )


def test_extract_candidate_patch_rejects_explanation_only():

    try:

        _extract_candidate_patch(
            "I would update the parser."
        )

    except ValueError as exc:

        assert (
            "unified source patch"
            in str(
                exc
            )
        )

    else:

        raise AssertionError(
            "Explanation-only output was accepted."
        )


def test_context_path_policy():

    assert (
        _context_path_allowed(
            "src/app.py"
        )
        is True
    )

    assert (
        _context_path_allowed(
            "lib/app.ex"
        )
        is True
    )

    assert (
        _context_path_allowed(
            ".git/config"
        )
        is False
    )

    assert (
        _context_path_allowed(
            "node_modules/pkg/a.js"
        )
        is False
    )

    assert (
        _context_path_allowed(
            "../outside.py"
        )
        is False
    )


def test_issue_keywords_remove_common_noise():

    result = (
        _issue_keywords(
            "Fix WidgetParser because the parser "
            "fails when empty_value is provided."
        )
    )

    normalized = {
        value.casefold()

        for value
        in result
    }

    assert (
        "widgetparser"
        in normalized
    )

    assert (
        "empty_value"
        in normalized
    )

    assert (
        "because"
        not in normalized
    )

    assert (
        "fix"
        not in normalized
    )


def test_context_excerpt_prefers_issue_match():

    text = (
        "x"
        * 8000
        + "\nWidgetParser broken here\n"
        + (
            "y"
            * 8000
        )
    )

    (
        excerpt,
        truncated,
    ) = (
        _context_excerpt(
            text,
            [
                "WidgetParser"
            ],
            max_chars=3000,
        )
    )

    assert (
        "WidgetParser"
        in excerpt
    )

    assert (
        truncated
        is True
    )


def test_static_patch_validator_accepts_normal_git_diff():

    from learning.evaluation.developer_candidate_patch import (
        _validate_candidate_patch_text,
    )

    patch = """diff --git a/a.py b/a.py
index 1111111..2222222 100644
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-old
+new
"""

    assert (
        _validate_candidate_patch_text(
            patch
        )
        == []
    )


def test_static_patch_validator_rejects_generated_malformed_headers():

    from learning.evaluation.developer_candidate_patch import (
        _validate_candidate_patch_text,
    )

    patch = """diff --git a/earthkit/data/readers/netcdf.py b/earthkit/data/readers/netcdf.py
index f21e226bdbea61fe35ea91c111e553c9e7ba59a0..f21e226bdbea61fe35ea91c111e553c9e1d9a59a0 --diff
rename index f21e226bdbea61fe35ea91c111e553c9e7ba59a0..f21e226bdbea61fe35ea91c111e553c91b7d9a59a0
--- a/earthkit/data/readers/netcdf.py
+++ b/earthkit/data/readers/netcdf.py
@@ -36,7 +36,7 @@ def as_level(self, level):
     n = float(level)
     if int(n) == n:
         return int(n)
-    return n
+    return level
"""

    errors = (
        _validate_candidate_patch_text(
            patch
        )
    )

    assert (
        "malformed_index_header:2"
        in errors
    )

    assert (
        "unsupported_extended_header:3"
        in errors
    )



def test_issue_keywords_prioritize_stack_trace_symbols():

    problem = """
Connection Issue: Authentication failed.

Traceback:
  File "/tmp/bimmer_connected/account.py", line 170, in _login_row_na
    response.raise_for_status()
"""

    result = (
        _issue_keywords(
            problem
        )
    )

    assert (
        "_login_row_na"
        in result[
            :4
        ]
    )

    assert (
        "raise_for_status"
        in result
    )


def test_issue_referenced_paths_resolve_absolute_traceback_path():

    tree = [
        ".readthedocs.yml",
        "bimmer_connected/account.py",
        "docs/source/development/reverse_engineering_mybmw.rst",
        "test/responses/auth/auth_error_wrong_password.json",
    ]

    problem = """
Traceback:
  File "/home/pi/.local/lib/python3.7/site-packages/bimmer_connected/account.py", line 170, in _login_row_na
    response.raise_for_status()
"""

    assert (
        _issue_referenced_paths(
            problem,
            tree,
        )
        == [
            "bimmer_connected/account.py"
        ]
    )


def test_referenced_source_path_outranks_generic_content_matches():

    tree = [
        ".readthedocs.yml",
        "bimmer_connected/account.py",
        "docs/source/development/reverse_engineering_mybmw.rst",
        "test/responses/auth/auth_error_wrong_password.json",
    ]

    ranked = (
        _rank_context_paths(
            tree=tree,

            matched=[
                ".readthedocs.yml",
                "docs/source/development/reverse_engineering_mybmw.rst",
                "test/responses/auth/auth_error_wrong_password.json",
            ],

            referenced=[
                "bimmer_connected/account.py"
            ],

            keywords=[
                "Authentication",
                "_login_row_na",
            ],
        )
    )

    assert (
        ranked[
            0
        ]
        == "bimmer_connected/account.py"
    )


def test_multiple_referenced_paths_preserve_issue_order():

    from learning.evaluation.developer_candidate_patch import (
        _rank_context_paths,
    )

    tree = [
        "bimmer_connected/account.py",
        "bimmer_connected/cli.py",
        "docs/source/development/reverse_engineering_mybmw.rst",
    ]

    ranked = (
        _rank_context_paths(
            tree=tree,

            matched=[
                "bimmer_connected/account.py",
                "bimmer_connected/cli.py",
            ],

            referenced=[
                "bimmer_connected/account.py",
                "bimmer_connected/cli.py",
            ],

            # Deliberately favor cli.py lexically. The explicit
            # traceback-reference order must still win.
            keywords=[
                "main",
                "fingerprint",
                "cli",
                "_login_row_na",
            ],
        )
    )

    assert (
        ranked[
            :2
        ]
        == [
            "bimmer_connected/account.py",
            "bimmer_connected/cli.py",
        ]
    )


def test_context_excerpt_prefers_dense_relevant_source_region():

    noise = (
        "package_name documentation import helper\\n"
        * 200
    )

    relevant = """
class Account:

    def _get_oauth_token(self):
        token_data = self._login_service()
        return token_data

    def _login_service(self):
        response = session.post(
            authenticate_url
        )
        response.raise_for_status()
        return response.json()
"""

    trailing = (
        "unrelated helper code\\n"
        * 200
    )

    text = (
        noise
        + relevant
        + trailing
    )

    excerpt, truncated = (
        _context_excerpt(
            text,
            [
                "_get_oauth_token",
                "_login_service",
                "raise_for_status",
                "package_name",
            ],
            max_chars=1600,
        )
    )

    assert (
        "_get_oauth_token"
        in excerpt
    )

    assert (
        "_login_service"
        in excerpt
    )

    assert (
        "raise_for_status"
        in excerpt
    )

    assert (
        truncated
        is True
    )
