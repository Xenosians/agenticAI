from subagents.core.tooling.gateway import resolve_policy_execution_arguments


def test_policy_binding_preserves_user_arguments():
    resolved, error = resolve_policy_execution_arguments(
        tool_name="example",
        tool={"trusted_policy_arguments": ["expected_version"]},
        original_arguments={"target": "alpha"},
        policy_result={"execution_arguments": {"target": "alpha", "expected_version": "v1"}},
    )
    assert error is None
    assert resolved == {"target": "alpha", "expected_version": "v1"}


def test_policy_binding_rejects_changed_user_argument():
    resolved, error = resolve_policy_execution_arguments(
        tool_name="example",
        tool={"trusted_policy_arguments": ["expected_version"]},
        original_arguments={"target": "alpha"},
        policy_result={"execution_arguments": {"target": "beta", "expected_version": "v1"}},
    )
    assert resolved is None
    assert "changed original argument" in error


def test_policy_binding_rejects_undeclared_hidden_argument():
    resolved, error = resolve_policy_execution_arguments(
        tool_name="example",
        tool={"trusted_policy_arguments": []},
        original_arguments={"target": "alpha"},
        policy_result={"execution_arguments": {"target": "alpha", "surprise": "nope"}},
    )
    assert resolved is None
    assert "undeclared trusted execution arguments" in error
