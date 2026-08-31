import json
import re

from .approvals import create_approval
from .llm import ask
from .mcp_client import mcp_runtime

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

6. When producing user_id, copy it EXACTLY from the user's request.
   Do not correct it.
   Do not abbreviate it.
   Do not change spelling.
   Do not remove characters.

7. Never invent tools.

8. Never invent tool results.

9. unlock_user and reset_password are mutating operations.

10. Mutating operations require human approval.

11. A proposed action has NOT been executed.

12. Never claim a mutating action succeeded unless execution
    occurred after approval.

13. Do not include anything before ACTION or FINAL.


EXAMPLES:

User:
Is jdoe locked?

Assistant:
ACTION: account_status
ARGS: {"user_id": "jdoe"}


User:
Is asmith locked?

Assistant:
ACTION: account_status
ARGS: {"user_id": "asmith"}


User:
Does jdoe have VPN access?

Assistant:
ACTION: check_access
ARGS: {"user_id": "jdoe", "resource": "VPN access"}


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


def _identifier_appears_in_request(
    identifier: str,
    user_input: str,
) -> bool:
    """
    Verify that an identifier produced by the LLM
    actually appears in the original user request.

    Case differences are allowed, but spelling changes
    are NOT allowed.

    Example:

        user input:
            "is asmith locked?"

        accepted:
            asmith
            ASMITH

        rejected:
            Amith
            asmit
            jsmith
    """

    pattern = (
        rf"(?<![A-Za-z0-9._-])"
        rf"{re.escape(identifier)}"
        rf"(?![A-Za-z0-9._-])"
    )

    return (
        re.search(
            pattern,
            user_input,
            re.IGNORECASE,
        )
        is not None
    )


def _validate_identifiers(
    user_input: str,
    arguments: dict,
) -> tuple[bool, str | None]:
    """
    Validate identity-sensitive tool arguments.

    Qwen is allowed to choose the operation,
    but it is NOT trusted to invent or alter identities.
    """

    user_id = arguments.get("user_id")

    if user_id is None:
        return True, None

    if not isinstance(user_id, str):
        return (
            False,
            "user_id must be a string.",
        )

    if not _identifier_appears_in_request(
        user_id,
        user_input,
    ):
        return (
            False,
            (
                f"The model produced user_id '{user_id}', "
                "but that identifier does not appear exactly "
                "in the original request."
            ),
        )

    return True, None


async def run_agent(
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

        print(
            f"\n[LLM]\n{response}"
        )

        action_match = ACTION_PATTERN.search(
            response
        )

        # -------------------------------------------------
        # TOOL CALL
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
                    "for tool arguments."
                )

            tool = get_tool(
                tool_name
            )

            if tool is None:
                return (
                    f"Unknown tool requested: "
                    f"{tool_name}"
                )

            # =============================================
            # IDENTITY GUARD
            #
            # Never allow the LLM to silently alter an
            # account identifier.
            # =============================================

            valid, validation_error = (
                _validate_identifiers(
                    user_input,
                    arguments,
                )
            )

            if not valid:
                print(
                    "\n[Identifier Guard]"
                )
                print(
                    validation_error
                )

                # Give the model ONE MORE opportunity to
                # copy the identifier correctly.
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
                            f"{validation_error}\n\n"
                            "Try again.\n"
                            "Copy user_id EXACTLY, "
                            "character-for-character, from "
                            "the ORIGINAL user request.\n"
                            "Do not guess or correct it.\n\n"
                            "Respond using ACTION and ARGS."
                        ),
                    }
                )

                continue

            print("\n[Tool Call]")
            print(f"Name: {tool_name}")
            print(f"Args: {arguments}")
            print(f"Risk: {tool['risk']}")

            # =============================================
            # MUTATING OPERATION
            # =============================================

            if tool["requires_approval"]:
                approval = create_approval(
                    tool_name,
                    arguments,
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
            # READ OPERATION
            # =============================================

            print("\n[MCP Call]")

            tool_result = (
                await mcp_runtime.call_tool(
                    tool_name,
                    arguments,
                )
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
                        "Tool result:\n"
                        f"{json.dumps(tool_result)}\n\n"
                        "Answer the ORIGINAL request "
                        "using only this tool result.\n\n"
                        "Respond exactly:\n"
                        "FINAL: your answer"
                    ),
                }
            )

            continue

        # -------------------------------------------------
        # FINAL RESPONSE
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
        "The agent could not produce a valid "
        "tool request after multiple attempts."
    )