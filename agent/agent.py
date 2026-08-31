import json
import re

from .approvals import create_approval
from .llm import ask
from .mcp_client import call_mcp_tool

from tools.registry import get_tool


SYSTEM_PROMPT = """
You are an ITSM agent.

Available tools:

account_status
- Check whether an account is enabled or locked.
- Arguments:
  - user_id: string

check_access
- Check whether a user has access to a resource.
- Arguments:
  - user_id: string
  - resource: string

unlock_user
- Request an account unlock.
- Arguments:
  - user_id: string

reset_password
- Request a password reset.
- Arguments:
  - user_id: string


TOOL CALL FORMAT:

ACTION: tool_name
ARGS: {"argument": "value"}


NORMAL RESPONSE FORMAT:

FINAL: your answer


RULES:

1. Use account_status for account status or lock questions.

2. Use check_access for access questions.

3. Use unlock_user for account unlock requests.

4. Use reset_password for password reset requests.

5. Never invent usernames.

6. Never invent tools.

7. Never invent tool results.

8. unlock_user and reset_password are mutating operations.

9. Mutating operations require human approval.

10. A proposed action has NOT been executed.

11. Never claim a mutating action succeeded unless actual
    execution occurred after approval.

12. Do not include anything before ACTION or FINAL.


EXAMPLES:

User:
Is jdoe locked?

Assistant:
ACTION: account_status
ARGS: {"user_id": "jdoe"}


User:
Does asmith have VPN access?

Assistant:
ACTION: check_access
ARGS: {"user_id": "asmith", "resource": "vpn"}


User:
Unlock asmith.

Assistant:
ACTION: unlock_user
ARGS: {"user_id": "asmith"}


User:
Reset jdoe's password.

Assistant:
ACTION: reset_password
ARGS: {"user_id": "jdoe"}
"""


ACTION_PATTERN = re.compile(
    r"ACTION:\s*([a-zA-Z0-9_-]+)"
    r"\s*ARGS:\s*(\{.*\})",
    re.DOTALL,
)

FINAL_PATTERN = re.compile(
    r"FINAL:\s*(.*)",
    re.DOTALL,
)


def run_agent(
    user_input: str,
) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    for _ in range(3):
        response = ask(messages)

        print(f"\n[LLM]\n{response}")

        action_match = ACTION_PATTERN.search(
            response
        )

        # -------------------------------------------------
        # TOOL REQUEST
        # -------------------------------------------------

        if action_match:
            tool_name = action_match.group(1)
            args_text = action_match.group(2)

            try:
                arguments = json.loads(
                    args_text
                )

            except json.JSONDecodeError:
                return (
                    "The model produced invalid JSON "
                    "for the tool arguments."
                )

            tool = get_tool(
                tool_name
            )

            if tool is None:
                return (
                    f"Unknown tool requested: {tool_name}"
                )

            print("\n[Tool Call]")
            print(f"Name: {tool_name}")
            print(f"Args: {arguments}")
            print(f"Risk: {tool['risk']}")

            # =============================================
            # MUTATING TOOL
            #
            # Do NOT send through MCP yet.
            # Create approval instead.
            # =============================================

            if tool["requires_approval"]:
                try:
                    approval = create_approval(
                        tool_name,
                        arguments,
                    )

                except Exception as exc:
                    return (
                        f"Failed to create approval: {exc}"
                    )

                print(
                    "\n[Approval Created]\n"
                    f"{approval}"
                )

                return (
                    f"Action '{tool_name}' requires "
                    f"human approval. "
                    f"Approval ID: {approval['id']} "
                    f"(risk: {approval['risk']})."
                )

            # =============================================
            # READ-ONLY TOOL
            #
            # This now goes through MCP.
            # =============================================

            print("\n[MCP Call]")

            tool_result = call_mcp_tool(
                tool_name,
                arguments,
            )

            print(
                "\n[MCP Result]\n"
                f"{tool_result}"
            )

            if not tool_result.get(
                "ok",
                False,
            ):
                return tool_result.get(
                    "error",
                    "MCP tool execution failed.",
                )

            # Give Qwen the actual MCP result.
            messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The read-only MCP tool "
                        "has been executed.\n\n"
                        "Tool result:\n"
                        f"{json.dumps(tool_result)}\n\n"
                        "Answer the ORIGINAL user request "
                        "using only this result.\n\n"
                        "Respond exactly as:\n"
                        "FINAL: your answer"
                    ),
                }
            )

            continue

        # -------------------------------------------------
        # NORMAL RESPONSE
        # -------------------------------------------------

        final_match = FINAL_PATTERN.search(
            response
        )

        if final_match:
            return final_match.group(
                1
            ).strip()

        return response.strip()

    return (
        "The agent exceeded the maximum "
        "number of steps."
    )