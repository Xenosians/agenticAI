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
    assert report.case_set is None and report.case_set_sha256 is None


def test_specialist_report_binds_the_scored_case_set():
    case = SpecialistModelCaseResult(
        name="get", user_request="Get KAN-3", expected_tool="ticket_get",
        expected_arguments={"ticket_key": "KAN-3"}, passed=True, duration_seconds=0.1,
    )
    reports = [
        finalize_specialist_model_report(
            agent_name="jira-specialist", model_key="jira-func", backend="hf-causal",
            quantization="bnb4", compute_dtype="bfloat16", device_map="auto", cases=[case],
            case_set=case_set, case_set_sha256=digest,
        )
        for case_set, digest in (("development", "a" * 64), ("holdout", "b" * 64))
    ]
    assert [report.case_set for report in reports] == ["development", "holdout"]
    assert reports[1].model_dump(by_alias=True)["case_set_sha256"] == "b" * 64
    assert reports[0].report_sha256 != reports[1].report_sha256
