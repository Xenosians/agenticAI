"""One full-catalog prompt contract for serving and offline evaluation/training."""
from subagents.core.orchestration.router_catalog import render_router_catalog
from subagents.prompts.prompt_loader import load_prompt


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
