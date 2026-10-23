"""One full-catalog prompt contract for serving and offline evaluation/training."""
from subagents.core.orchestration.router_catalog import render_router_catalog
from subagents.prompts.prompt_loader import load_prompt


def build_compact_router_system_prompt(
    specialists: list[dict], *, include_workflow_protocol: bool = True,
) -> str:
    """Render trusted catalog metadata under the compact semantic contract."""
    system = load_prompt("hub_router_compact.txt").replace(
        "{{SPECIALISTS_JSON}}", render_router_catalog(specialists),
    ).replace(
        "{{WORKFLOW_PROTOCOL}}",
        load_prompt("hub_workflow_compact.txt") if include_workflow_protocol
        else "Conditional workflow generation is disabled for this repair.",
    )
    return system + "\n\n" + load_prompt("hub_success_continuation.txt")


def build_router_system_prompt(
    specialists: list[dict], *, repair_mode: bool = False,
    include_workflow_protocol: bool = True,
) -> str:
    parts = [load_prompt("hub_router.txt").replace(
        "{{SPECIALISTS_JSON}}", render_router_catalog(specialists),
    )]
    if include_workflow_protocol:
        parts.append(load_prompt("hub_conditional_workflow.txt"))
    if repair_mode:
        parts.append(load_prompt("hub_router_repair.txt"))
    # Mutation-success ordering is distinct from a read-result predicate.
    # Keep its exact field nesting available during bounded repair too.
    parts.append(load_prompt("hub_success_continuation.txt"))
    return "\n\n".join(parts)
