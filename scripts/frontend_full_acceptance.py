#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print(
        "Playwright missing.\n"
        "Install with:\n"
        "  python -m pip install playwright\n"
        "  python -m playwright install chromium",
        file=sys.stderr,
    )
    raise SystemExit(2)

from scripts.frontend_e2e_acceptance import (
    BrowserEvidence,
    TERMINAL,
    login,
    load_cases,
    run_case,
    new_chat,
    send_prompt,
)

from scripts.tool_acceptance_matrix import (
    load_inventory,
    load_proposal_cases,
    validate_case,
)


PLATFORM_MANIFEST = ROOT / "config" / "cross_repo_acceptance_cases.json"
REPORT_ROOT = ROOT / ".runtime" / "acceptance" / "frontend-full"


def now_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def contract_dict(contract) -> dict[str, Any]:
    value = contract.as_dict()

    if not isinstance(value, dict):
        raise RuntimeError("Capability contract did not serialize to a dict.")

    return value


def choose_tool_cases():
    contracts, agents = load_inventory()
    cases = load_proposal_cases()

    priority = {
        "existing_eval": 0,
        "reviewed": 1,
        "authored_unreviewed": 2,
    }

    selected = {}

    for case in cases:
        if case.training_overlap:
            continue

        if validate_case(case, contracts, agents) is not None:
            continue

        tool = case.expected_tool

        if tool not in contracts:
            continue

        current = selected.get(tool)

        if current is None:
            selected[tool] = case
            continue

        old_rank = priority.get(current.review_status, 99)
        new_rank = priority.get(case.review_status, 99)

        if new_rank < old_rank:
            selected[tool] = case

    missing = sorted(set(contracts) - set(selected))

    if missing:
        raise RuntimeError(
            "No usable FE proposal case for: " + ", ".join(missing)
        )

    return contracts, agents, selected


def seen_tool(
    evidence: BrowserEvidence,
    job_id: str,
    expected_tool: str,
) -> bool:
    for item in evidence.job_snapshots.get(job_id, []):
        if item.get("tool") == expected_tool:
            return True

    return False


def latest_snapshot(
    evidence: BrowserEvidence,
    job_id: str,
) -> dict[str, Any] | None:
    values = evidence.job_snapshots.get(job_id, [])
    return values[-1] if values else None


def wait_for_any(
    page,
    evidence: BrowserEvidence,
    job_id: str,
    statuses: set[str],
    timeout_ms: int,
) -> dict[str, Any]:
    return evidence.wait_status(
        page,
        job_id,
        statuses,
        timeout_ms,
    )


def run_tool_case(
    page,
    evidence: BrowserEvidence,
    *,
    tool_name: str,
    contract,
    proposal_case,
    timeout_ms: int,
) -> dict[str, Any]:

    contract_data = contract_dict(contract)

    requires_approval = bool(
        contract_data.get("requires_approval", False)
    )

    effect = str(
        contract_data.get("effect", "")
    )

    record = {
        "tool": tool_name,
        "specialist": proposal_case.agent_name,
        "case_id": proposal_case.case_id,
        "review_status": proposal_case.review_status,
        "effect": effect,
        "requires_approval": requires_approval,
        "verdict": "FAIL",
    }

    evidence.current_case = "tool:" + tool_name

    new_chat(
        page,
        timeout_ms,
    )

    job_id, submit = send_prompt(
        page,
        proposal_case.user,
        timeout_ms,
    )

    record["job_id"] = job_id
    record["submission_status"] = submit.get("status")

    if requires_approval:
        state = wait_for_any(
            page,
            evidence,
            job_id,
            {"waiting_approval"} | TERMINAL,
            timeout_ms,
        )

        record["pre_decision"] = state

        if state["status"] != "waiting_approval":
            record["verdict"] = "FAIL_SECURITY"

            record["reason"] = (
                "Approval-required capability reached "
                f"{state['status']!r} without waiting_approval."
            )

            return record

        if state.get("tool") != tool_name:
            record["verdict"] = "FAIL_ROUTING"

            record["reason"] = (
                f"Expected {tool_name!r}, observed "
                f"{state.get('tool')!r}."
            )

            return record

        approval_id = state.get("approval_id")

        if not approval_id:
            record["verdict"] = "FAIL_SECURITY"
            record["reason"] = "waiting_approval has no approval identity."
            return record

        # Full tool sweep is mutation-safe by default:
        # prove exact proposal + approval boundary, then REJECT.
        reject = page.get_by_role(
            "button",
            name="Reject action",
            exact=True,
        )

        reject.wait_for(
            state="visible",
            timeout=timeout_ms,
        )

        reject.click()

        final = wait_for_any(
            page,
            evidence,
            job_id,
            {"denied", "failed"},
            timeout_ms,
        )

        record["final"] = final

        if final["status"] != "denied":
            record["verdict"] = "FAIL_REJECTION"
            record["reason"] = (
                "Rejected proposal did not persist as denied."
            )
            return record

        # Durable reload test: the rejected job must not become actionable again.
        page.reload(wait_until="domcontentloaded")

        page.locator("#messageInput").wait_for(
            state="visible",
            timeout=timeout_ms,
        )

        page.wait_for_timeout(500)

        if page.get_by_role(
            "button",
            name="Approve action",
            exact=True,
        ).count():
            record["verdict"] = "FAIL_RELOAD_AUTHORITY"
            record["reason"] = (
                "Rejected mutation restored an approval button after reload."
            )
            return record

        record["reload_persistence"] = "PASS"
        record["verdict"] = "PASS_PROPOSAL_REJECTED"
        return record

    # Non-approval capabilities are allowed to execute, because these should be
    # non-mutating/read-style capabilities.
    state = wait_for_any(
        page,
        evidence,
        job_id,
        TERMINAL,
        timeout_ms,
    )

    record["final"] = state

    correct_tool = seen_tool(
        evidence,
        job_id,
        tool_name,
    )

    record["expected_tool_observed"] = correct_tool

    if state["status"] == "completed" and correct_tool:
        record["verdict"] = "PASS_EXECUTED"
        return record

    if state["status"] == "completed" and not correct_tool:
        record["verdict"] = "FAIL_ROUTING"
        record["reason"] = (
            "Job completed but the expected tool was never observed."
        )
        return record

    if correct_tool:
        record["verdict"] = "FAIL_PROVIDER_OR_EXECUTION"

        record["reason"] = (
            "Correct tool was reached, but durable execution ended as "
            f"{state['status']!r}."
        )

        return record

    record["verdict"] = "FAIL_ROUTING"

    record["reason"] = (
        "Expected tool was not observed before terminal state "
        f"{state['status']!r}."
    )

    return record


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Frontend Full Acceptance",
        "",
        f"- Created: `{report['created_at']}`",
        f"- Overall: **{report['summary']['overall']}**",
        f"- Registered tools: **{report['summary']['registered_tools']}**",
        f"- Tool cases executed: **{report['summary']['tool_cases']}**",
        f"- Platform cases: **{report['summary']['platform_cases']}**",
        "",
        "## Tool results",
        "",
        "| Tool | Effect | Approval | Verdict |",
        "|---|---|---:|---|",
    ]

    for item in report["tools"]:
        lines.append(
            "| `{}` | `{}` | {} | **{}** |".format(
                item["tool"],
                item.get("effect", ""),
                "yes" if item.get("requires_approval") else "no",
                item["verdict"],
            )
        )

    lines += [
        "",
        "## Platform results",
        "",
        "| Case | Result |",
        "|---|---|",
    ]

    for item in report["platform"]:
        lines.append(
            f"| `{item['id']}` | **{item['status']}** |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Drive platform acceptance and every registered capability "
            "through the real browser frontend."
        )
    )

    parser.add_argument(
        "--frontend-url",
        default=os.environ.get(
            "FRONTEND_URL",
            "http://127.0.0.1:8080",
        ),
    )

    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show Chromium while the suite runs.",
    )

    parser.add_argument(
        "--force-login",
        action="store_true",
        help="Clear cookies and prove fresh frontend authentication.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--platform-only",
        action="store_true",
    )

    parser.add_argument(
        "--tools-only",
        action="store_true",
    )

    parser.add_argument(
        "--tool",
        action="append",
        default=[],
        help="Run only specific tool(s). Repeatable.",
    )

    parser.add_argument(
        "--include-integration",
        action="append",
        default=[],
        choices=[
            "openwrt",
        ],
    )

    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        default=True,
    )

    args = parser.parse_args()

    if args.platform_only and args.tools_only:
        parser.error("--platform-only and --tools-only are mutually exclusive")

    run_platform = not args.tools_only
    run_tools = not args.platform_only

    timeout_ms = int(args.timeout * 1000)
    token = now_token()

    report_dir = REPORT_ROOT / token
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "schema": "frontend-full-acceptance.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frontend_url": args.frontend_url,
        "platform": [],
        "tools": [],
        "summary": {},
    }

    contracts, agents, tool_cases = choose_tool_cases()

    requested_tools = set(args.tool)

    if requested_tools:
        unknown = sorted(requested_tools - set(contracts))

        if unknown:
            raise SystemExit(
                "Unknown tool(s): " + ", ".join(unknown)
            )

    selected_tools = [
        name
        for name in sorted(contracts)
        if not requested_tools or name in requested_tools
    ]

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=not args.headed
        )

        context = browser.new_context()
        page = context.new_page()

        evidence = BrowserEvidence()
        evidence.attach(page)

        try:
            report["auth"] = login(
                page,
                args.frontend_url,
                timeout_ms,
                force_login=args.force_login,
            )

            if run_platform:
                platform_cases = load_cases(
                    PLATFORM_MANIFEST,
                    [],
                )

                for case in platform_cases:
                    print(f"[PLATFORM] {case['id']}")

                    try:
                        result = run_case(
                            page,
                            evidence,
                            case,
                            run_token=token,
                            timeout_ms=timeout_ms,
                            execute_disposable=False,
                            workspace_root=Path(
                                os.environ.get(
                                    "ACCEPTANCE_WORKSPACE_ROOT",
                                    "/mnt/c/project",
                                )
                            ),
                            integrations=set(
                                args.include_integration
                            ),
                        )

                    except Exception as exc:
                        screenshot = (
                            report_dir
                            / f"platform-{case['id']}.png"
                        )

                        try:
                            page.screenshot(
                                path=str(screenshot),
                                full_page=True,
                            )
                        except Exception:
                            pass

                        result = {
                            "id": case["id"],
                            "status": "FAIL",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "screenshot": str(screenshot),
                        }

                    finally:
                        evidence.current_case = None

                    report["platform"].append(result)

                    print(
                        f"  -> {result['status']}"
                        + (
                            f" · {result.get('error')}"
                            if result.get("error")
                            else ""
                        )
                    )

            if run_tools:
                for index, tool_name in enumerate(
                    selected_tools,
                    1,
                ):
                    print(
                        f"[TOOL {index:02d}/{len(selected_tools):02d}] "
                        f"{tool_name}"
                    )

                    case = tool_cases[tool_name]

                    try:
                        item = run_tool_case(
                            page,
                            evidence,
                            tool_name=tool_name,
                            contract=contracts[tool_name],
                            proposal_case=case,
                            timeout_ms=timeout_ms,
                        )

                    except Exception as exc:
                        screenshot = (
                            report_dir
                            / f"tool-{tool_name}.png"
                        )

                        try:
                            page.screenshot(
                                path=str(screenshot),
                                full_page=True,
                            )
                        except Exception:
                            pass

                        item = {
                            "tool": tool_name,
                            "case_id": case.case_id,
                            "review_status": case.review_status,
                            "verdict": "FAIL_EXCEPTION",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "screenshot": str(screenshot),
                        }

                    finally:
                        evidence.current_case = None

                    report["tools"].append(item)

                    print(
                        "  -> "
                        + item["verdict"]
                        + (
                            " · " + item["reason"]
                            if item.get("reason")
                            else ""
                        )
                    )

        finally:
            context.close()
            browser.close()

    platform_counts = Counter(
        item["status"]
        for item in report["platform"]
    )

    tool_counts = Counter(
        item["verdict"]
        for item in report["tools"]
    )

    platform_fail = platform_counts["FAIL"]

    tool_fail = sum(
        count
        for verdict, count in tool_counts.items()
        if verdict.startswith("FAIL")
    )

    report["summary"] = {
        "registered_tools": len(contracts),
        "tool_cases": len(report["tools"]),
        "platform_cases": len(report["platform"]),
        "platform_counts": dict(platform_counts),
        "tool_counts": dict(tool_counts),
        "websocket_observed": bool(evidence.websockets),
        "socket_token_200_observed": any(
            item.get("status") == 200
            for item in evidence.socket_tokens
        ),
        "overall": (
            "PASS"
            if platform_fail == 0 and tool_fail == 0
            else "FAIL"
        ),
    }

    json_path = report_dir / "report.json"
    md_path = report_dir / "report.md"

    json_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    md_path.write_text(
        markdown_report(report),
        encoding="utf-8",
    )

    print()
    print("FULL FRONTEND ACCEPTANCE")
    print("========================")
    print(f"registered tools : {len(contracts)}")
    print(f"tool cases       : {len(report['tools'])}")
    print(f"platform cases   : {len(report['platform'])}")
    print()

    for key in sorted(tool_counts):
        print(f"{key:28} {tool_counts[key]}")

    if report["platform"]:
        print()

        for key in sorted(platform_counts):
            print(f"platform {key:19} {platform_counts[key]}")

    print()
    print(
        "websocket observed : "
        + (
            "yes"
            if report["summary"]["websocket_observed"]
            else "no"
        )
    )

    print(
        "socket-token 200   : "
        + (
            "yes"
            if report["summary"]["socket_token_200_observed"]
            else "no"
        )
    )

    print(f"overall            : {report['summary']['overall']}")
    print(f"json report        : {json_path}")
    print(f"markdown report    : {md_path}")

    return (
        0
        if report["summary"]["overall"] == "PASS"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
