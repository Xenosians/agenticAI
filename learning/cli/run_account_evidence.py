from __future__ import annotations

import argparse
import asyncio
import json
import uuid

from pathlib import (
    Path,
)

from agent.mcp_client import (
    MCPRuntime,
)

from config import (
    Settings,
)

from learning.evidence.account_exercise import (
    AccountEvidenceMcpGuard,
    AccountEvidenceObservation,
    build_account_evidence_observation,
    build_account_evidence_session_report,
    create_account_evidence_approval,
    write_account_evidence_session_report,
)

from learning.evidence.recorder import (
    TrajectoryRecorder,
)

from learning.paths import (
    RUNTIME_LEARNING_ROOT,
    TRAJECTORIES_PATH,
)

from subagents.core.tooling.gateway import (
    ToolGateway,
)

from subagents.hub import (
    build_hub,
)

from subagents.llm.runtime.inference import (
    InferenceCoordinator,
)

from subagents.llm.runtime.model_manager import (
    ModelManager,
)

from subagents.llm.runtime.scheduler import (
    GpuScheduler,
)


DEFAULT_TRAJECTORIES = (
    TRAJECTORIES_PATH
)


DEFAULT_REPORT_ROOT = (
    RUNTIME_LEARNING_ROOT
    / "evidence-runs"
    / "account"
)


# ============================================================
# CLI
# ============================================================


def build_parser(
) -> argparse.ArgumentParser:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Run user-supplied requests through the real "
                "Hub/router/account-specialist runtime and capture "
                "append-only provenance-bearing learning "
                "trajectories without creating corrections, "
                "reviews, datasets, or executable approvals."
            )
        )
    )

    parser.add_argument(
        "--request",
        dest="requests",
        action="append",
        required=True,
        help=(
            "Natural-language request to execute. "
            "May be supplied multiple times."
        ),
    )

    parser.add_argument(
        "--trajectories",
        type=Path,
        default=(
            DEFAULT_TRAJECTORIES
        ),
    )

    parser.add_argument(
        "--report-root",
        type=Path,
        default=(
            DEFAULT_REPORT_ROOT
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print the complete derived session report as JSON."
        ),
    )

    return parser


# ============================================================
# DISPLAY
# ============================================================


def _print_observation(
    observation: AccountEvidenceObservation,
) -> None:

    print()
    print(
        f"[{observation.request_index}] "
        f"{observation.user_request}"
    )

    print(
        "  trajectory =",
        observation.trajectory_id,
    )

    print(
        "  hub status =",
        observation.hub_status,
    )

    print(
        "  routes     =",
        (
            ", ".join(
                observation.routes
            )
            if observation.routes
            else "(none)"
        ),
    )

    print(
        "  account    =",
        (
            "yes"
            if observation.account_routed
            else "no"
        ),
    )

    if not observation.account_results:

        print(
            "  account result = (none)"
        )

        return

    for (
        result_index,
        result,
    ) in enumerate(
        observation.account_results,
        start=1,
    ):

        print(
            f"  account result #{result_index}"
        )

        print(
            "    task_id       =",
            result.task_id,
        )

        print(
            "    status        =",
            result.status,
        )

        print(
            "    outcome       =",
            result.outcome_code,
        )

        print(
            "    proposed tool =",
            result.proposed_tool,
        )

        print(
            "    arguments     =",
            result.proposed_arguments,
        )

        print(
            "    provenance    =",
            result.provenance_present,
        )

        print(
            "    prov complete =",
            result.provenance_complete,
        )

        print(
            "    model_key     =",
            result.model_key,
        )


# ============================================================
# RUNTIME
# ============================================================


async def _run(
    args,
) -> int:

    normalized_requests = [
        value.strip()

        for value
        in args.requests

        if value.strip()
    ]

    if not normalized_requests:

        raise ValueError(
            "At least one non-empty --request is required."
        )

    session_id = (
        "account-evidence-"
        f"{uuid.uuid4().hex}"
    )

    settings = (
        Settings()
    )

    raw_mcp = (
        MCPRuntime()
    )

    guarded_mcp = (
        AccountEvidenceMcpGuard(
            raw_mcp
        )
    )

    model_manager = (
        ModelManager(
            settings=(
                settings
            )
        )
    )

    scheduler = (
        GpuScheduler()
    )

    inference = (
        InferenceCoordinator(
            model_manager=(
                model_manager
            ),

            scheduler=(
                scheduler
            ),
        )
    )

    tool_gateway = (
        ToolGateway(
            approval_creator=(
                create_account_evidence_approval
            ),

            mcp=(
                guarded_mcp
            ),
        )
    )

    recorder = (
        TrajectoryRecorder(
            path=(
                args.trajectories
            ),

            enabled=True,

            hub_model=(
                settings.hub_model_key
            ),
        )
    )

    observations: list[
        AccountEvidenceObservation
    ] = []

    try:

        await raw_mcp.start()

        print(
            "[EVIDENCE] Warming Hub model..."
        )

        await inference.warm(
            settings.hub_model_key
        )

        hub = (
            build_hub(
                settings=(
                    settings
                ),

                model_manager=(
                    model_manager
                ),

                inference=(
                    inference
                ),

                tool_gateway=(
                    tool_gateway
                ),
            )
        )

        print(
            "[EVIDENCE] Runtime ready."
        )

        print(
            "[EVIDENCE] Mutation execution is "
            "fail-closed in this harness."
        )

        for (
            index,
            user_request,
        ) in enumerate(
            normalized_requests,
            start=1,
        ):

            job_id = (
                f"{session_id}-"
                f"{index:03d}"
            )

            print()
            print(
                "[EVIDENCE] Running "
                f"{index}/"
                f"{len(normalized_requests)}"
            )

            print(
                "[EVIDENCE] Request: "
                f"{user_request}"
            )

            result = (
                await hub.run(
                    user_request
                )
            )

            trajectory_payload = (
                recorder.record(
                    job_id=(
                        job_id
                    ),

                    attempt=1,

                    result=(
                        result
                    ),
                )
            )

            if trajectory_payload is None:

                raise RuntimeError(
                    "Trajectory recorder unexpectedly "
                    "returned no payload."
                )

            observation = (
                build_account_evidence_observation(
                    request_index=(
                        index
                    ),

                    job_id=(
                        job_id
                    ),

                    result=(
                        result
                    ),

                    trajectory_payload=(
                        trajectory_payload
                    ),
                )
            )

            observations.append(
                observation
            )

            if not args.json:

                _print_observation(
                    observation
                )

    finally:

        try:

            unloaded = (
                model_manager
                .unload_all()
            )

            if unloaded:

                print(
                    "[EVIDENCE] Unloaded models: "
                    + ", ".join(
                        unloaded
                    )
                )

        finally:

            await raw_mcp.stop()

    report = (
        build_account_evidence_session_report(
            session_id=(
                session_id
            ),

            observations=(
                observations
            ),
        )
    )

    report_path = (
        write_account_evidence_session_report(
            report=(
                report
            ),

            root=(
                args.report_root
            ),
        )
    )

    if args.json:

        print(
            report.model_dump_json(
                by_alias=True,
                indent=2,
            )
        )

    else:

        print()
        print(
            "Account Evidence Session"
        )

        print(
            "========================"
        )

        print(
            "Session ID:       ",
            report.session_id,
        )

        print(
            "Requests:         ",
            report.request_count,
        )

        print(
            "Account routed:   ",
            report.account_routed_count,
        )

        print(
            "Trajectory count: ",
            len(
                report.trajectory_ids
            ),
        )

        print(
            "Report:           ",
            report_path,
        )

        print()
        print(
            "No corrections were created."
        )

        print(
            "No reviews were created."
        )

        print(
            "No dataset was promoted."
        )

        print(
            "No mutating account tool was "
            "physically executed by this harness."
        )

    return 0


def main(
) -> int:

    args = (
        build_parser()
        .parse_args()
    )

    try:

        return (
            asyncio.run(
                _run(
                    args
                )
            )
        )

    except KeyboardInterrupt:

        print()
        print(
            "Interrupted."
        )

        return 130

    except Exception as exc:

        print(
            "ERROR: "
            f"{exc}"
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
