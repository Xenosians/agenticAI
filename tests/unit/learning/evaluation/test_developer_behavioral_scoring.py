from learning.evaluation.developer_behavioral_scoring import (
    _score_statuses,
    normalize_test_name,
    parse_log_elixir,
    parse_log_pytest,
)


def test_pytest_parser_matches_benchmark_format():

    log = """
PASSED tests/test_cli.py::test_ok
FAILED tests/test_cli.py::test_bad - AssertionError
SKIPPED tests/test_cli.py::test_skip
"""

    assert (
        parse_log_pytest(
            log
        )
        == {
            "tests/test_cli.py::test_ok":
                "PASSED",

            "tests/test_cli.py::test_bad":
                "FAILED",

            "tests/test_cli.py::test_skip":
                "SKIPPED",
        }
    )


def test_elixir_parser_tracks_pass_fail_and_skip():

    log = """
* test succeeds (1.2ms) [L#10]
* test skipped (skipped) [L#11]
* test later fails [L#12]
1) test later fails (Example.Test)
"""

    assert (
        parse_log_elixir(
            log
        )
        == {
            "succeeds":
                "PASSED",

            "skipped":
                "SKIPPED",

            "later fails":
                "FAILED",
        }
    )


def test_timing_normalization():

    assert (
        normalize_test_name(
            "test_a [12.4 ms]"
        )
        == "test_a"
    )

    assert (
        normalize_test_name(
            "test_a in 29.08 msec"
        )
        == "test_a"
    )

    assert (
        normalize_test_name(
            "test_a (123ms)"
        )
        == "test_a"
    )


def test_score_requires_all_fail_to_pass_and_pass_to_pass():

    (
        resolved,
        unresolved,
        preserved,
        regressed,
    ) = (
        _score_statuses(
            statuses={
                "bug_one":
                    "PASSED",

                "bug_two":
                    "FAILED",

                "existing_one":
                    "PASSED",

                "existing_two":
                    "FAILED",
            },

            fail_to_pass=[
                "bug_one",
                "bug_two",
            ],

            pass_to_pass=[
                "existing_one",
                "existing_two",
            ],
        )
    )

    assert (
        resolved
        == [
            "bug_one"
        ]
    )

    assert (
        unresolved
        == [
            "bug_two"
        ]
    )

    assert (
        preserved
        == [
            "existing_one"
        ]
    )

    assert (
        regressed
        == [
            "existing_two"
        ]
    )


def test_missing_expected_test_fails_closed():

    (
        resolved,
        unresolved,
        preserved,
        regressed,
    ) = (
        _score_statuses(
            statuses={},

            fail_to_pass=[
                "bug"
            ],

            pass_to_pass=[
                "regression"
            ],
        )
    )

    assert resolved == []

    assert unresolved == [
        "bug"
    ]

    assert preserved == []

    assert regressed == [
        "regression"
    ]
