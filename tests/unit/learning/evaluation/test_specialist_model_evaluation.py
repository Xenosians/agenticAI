from learning.evaluation.specialist_model import (
    SpecialistModelCaseResult,
    finalize_specialist_model_report,
)


def test_specialist_candidate_report_requires_all_cases_for_promotion_gate():
    report = (
        finalize_specialist_model_report(
            agent_name="jira-specialist",
            model_key="jira-func",
            backend="hf-causal",
            quantization="bnb4",
            compute_dtype="bfloat16",
            device_map="auto",
            cases=[
                SpecialistModelCaseResult(
                    name="get",
                    user_request="Get KAN-3",
                    expected_tool="ticket_get",
                    expected_arguments={
                        "ticket_key": "KAN-3",
                    },
                    observed_tool="ticket_get",
                    observed_arguments={
                        "ticket_key": "KAN-3",
                    },
                    raw_output="{}",
                    passed=True,
                    duration_seconds=0.1,
                ),
                SpecialistModelCaseResult(
                    name="assign",
                    user_request="Assign KAN-3",
                    expected_tool="ticket_assign",
                    expected_arguments={
                        "ticket_key": "KAN-3",
                        "assignee": "alice@example.com",
                    },
                    observed_tool="ticket_get",
                    observed_arguments={
                        "ticket_key": "KAN-3",
                    },
                    raw_output="{}",
                    passed=False,
                    duration_seconds=0.2,
                ),
            ],
        )
    )

    assert report.total == 2
    assert report.passed == 1
    assert report.failed == 1
    assert report.pass_rate == 0.5
    assert report.promotion_gate_passed is False
    assert len(report.report_sha256) == 64
