from __future__ import annotations

from pathlib import (
    Path,
)

from typing import (
    Any,
)

from learning.context import (
    StructuredContextRecord,
)

from learning.evidence.approval_lineage import (
    find_approval_lineage,
)

from learning.integrations.runtime_hooks import (
    ContinualLearningRuntimeHooks,
)


def capture_approval_execution_evidence(
    *,
    hooks: ContinualLearningRuntimeHooks,
    trajectory_path: Path,
    approval_id: str,
    approval_result: dict[
        str,
        Any,
    ],
) -> list[
    StructuredContextRecord
]:
    """
    Capture actual post-approval execution evidence and connect it
    to the original proposal trajectory.

    Replayed approvals deliberately create no duplicate learning
    records.
    """

    if not isinstance(
        approval_result,
        dict,
    ):

        return []

    if (
        approval_result.get(
            "replayed"
        )
        is True
    ):

        return []

    approval = (
        approval_result.get(
            "approval"
        )
    )

    if not isinstance(
        approval,
        dict,
    ):

        return []

    persisted_approval_id = (
        approval.get(
            "id"
        )
    )

    if (
        not isinstance(
            persisted_approval_id,
            str,
        )
        or persisted_approval_id
        != approval_id
    ):

        return []

    lineage = (
        find_approval_lineage(
            path=(
                trajectory_path
            ),

            approval_id=(
                approval_id
            ),
        )
    )

    if lineage is None:

        # Strict lineage:
        # do not create orphan mutation-learning evidence.
        return []

    provider_result = (
        approval_result.get(
            "result"
        )
    )

    if not isinstance(
        provider_result,
        dict,
    ):

        provider_result = None

    return (
        hooks.record_approval_execution(
            approval=(
                approval
            ),

            provider_result=(
                provider_result
            ),

            lineage=(
                lineage
            ),
        )
    )
