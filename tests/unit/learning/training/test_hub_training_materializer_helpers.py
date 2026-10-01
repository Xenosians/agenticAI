from learning.training.hub_training_materializer import (
    _assert_lineage_isolation,
)


def test_lineage_split_isolation_accepts_grouped_records():
    _assert_lineage_isolation(
        [
            {
                "lineage_id": "lesson:a",
                "partition": "train",
            },
            {
                "lineage_id": "lesson:a",
                "partition": "train",
            },
            {
                "lineage_id": "lesson:b",
                "partition": "validation",
            },
        ]
    )


def test_lineage_split_isolation_rejects_leakage():
    try:
        _assert_lineage_isolation(
            [
                {
                    "lineage_id": "trajectory:x",
                    "partition": "train",
                },
                {
                    "lineage_id": "trajectory:x",
                    "partition": "validation",
                },
            ]
        )

    except ValueError as exc:
        assert "leakage" in str(exc).lower()

    else:
        raise AssertionError(
            "Lineage leakage was accepted."
        )
