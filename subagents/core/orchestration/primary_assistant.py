import json
import os
import re
from collections.abc import Callable
from typing import Any

from config import (
    get_settings,
)

from subagents.llm.runtime.stages import InferenceStage

from services.capability_context import capability_messages
from tools.registry import get_tool
from services.extensions.context import answer_token_limit

from subagents.core.orchestration.conversation_context import (
    conversation_messages,
    prior_result_messages,
)

from subagents.core.definitions.types import (
    AgentResult,
)

from subagents.llm.runtime.inference import (
    InferenceEngine,
)

from subagents.llm.runtime.scheduler import (
    InferencePriority,
)

from subagents.prompts.prompt_loader import (
    load_prompt,
)


# Fields a user-facing answer can draw on. Provenance, routing contracts and UI
# presentation are audit/rendering data that only consume the synthesis budget.
SYNTHESIS_FIELDS = (
    "agent_name",
    "status",
    "proposed_tool",
    "proposed_arguments",
    "tool_result",
    "outcome_code",
    "error",
)


def answer_word_limit(max_new_tokens: int) -> int:
    """Words that fit an output cap: formatted answers cost two to three tokens
    per word, so a quarter of the cap keeps an answer from being cut off."""
    return max(40, max_new_tokens // 4)


def synthesis_evidence(result: AgentResult) -> dict:
    evidence = {
        field: getattr(result, field)
        for field in SYNTHESIS_FIELDS
        if getattr(result, field) is not None
    }
    tool = get_tool(result.proposed_tool) if result.proposed_tool else None
    answer_is_reading = (tool or {}).get("synthesis_evidence") == "answer"
    if result.answer and (result.tool_result is None or answer_is_reading):
        # Without a structured result the specialist's answer is the evidence.
        # A tool whose formatter states its complete reading in words declares
        # synthesis_evidence="answer": its raw flags are not reinterpreted.
        evidence.pop("tool_result", None)
        evidence["answer"] = result.answer
    return evidence


def complete_reading(results: list[AgentResult]) -> str | None:
    """The answer itself when one successful read's tool declares its reading complete and it is one line.

    Such a reading is already a direct statement; a synthesis model can only
    restate it or add to it, and live it added a wrong definition and a next
    step. A multi-line reading is a listing that synthesis narrows to the
    question, so it still goes through the model.
    """
    if len(results) != 1:
        return None
    result = results[0]
    tool = get_tool(result.proposed_tool) if result.proposed_tool else None
    answer = (result.answer or "").strip()
    if (result.status != "success" or (tool or {}).get("synthesis_evidence") != "answer"
            or not answer or "\n" in answer):
        return None
    return answer


# A standalone figure in a reading: a number not glued to a name (nic0 and vmbr0 are names).
_FIGURE = re.compile(r"(?<![A-Za-z_])\d+(?:[.,:]\d+)*")
# A complete reading with at least this many figures is shown exactly (owner's rule, 16 November).
MANY_FIGURES = 6
# Words that state a quantity. "one" stays allowed: it is mostly a determiner ("one short peak").
_NUMBER_WORDS = re.compile(
    r"\b(?:zero|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundreds?|"
    r"thousands?|millions?|billions?|half|halves|quarters?|dozens?|percent|percentage|twice|thrice)\b",
    re.IGNORECASE,
)
SUMMARY_MAX_CHARACTERS = 280
SUMMARY_MAX_NEW_TOKENS = 120


def reading_with_many_figures(results: list[AgentResult]) -> str | None:
    """The reading itself when one successful read's tool declares it complete and it is full of figures.

    Live, synthesis restated the node-load reading with a peak given as an
    average and the least available memory as an average. Such a reading is
    shown exactly as the tool wrote it; the model adds only a figure-free
    summary. One-line readings are already the answer (complete_reading).
    """
    if len(results) != 1:
        return None
    result = results[0]
    tool = get_tool(result.proposed_tool) if result.proposed_tool else None
    answer = (result.answer or "").strip()
    if (result.status != "success" or (tool or {}).get("synthesis_evidence") != "answer"
            or "\n" not in answer or len(_FIGURE.findall(answer)) < MANY_FIGURES):
        return None
    return answer


def plain_summary(text: str) -> str | None:
    """One or two sentences with no digit, no number word and no percent sign, or None."""
    summary = " ".join((text or "").split())
    if not summary or len(summary) > SUMMARY_MAX_CHARACTERS:
        return None
    if any(character.isdigit() for character in summary) or "%" in summary or _NUMBER_WORDS.search(summary):
        return None
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", summary) if part]
    return summary if len(sentences) <= 2 else None


# What a reading says it does not know. Live (18 November), asked whether there
# were pending package updates, a summary said "There are no pending package
# updates available." above a reading that says they are not read. An answer
# may name such a subject only to say it is not known.
_UNREADABLE_LIST = re.compile(r"not readable with janus's access:\s*([^.\n]+)", re.IGNORECASE)
_UNKNOWN_CLAUSE = re.compile(
    r"^(?P<subject>.+?)\s+(?:(?:is|are|was|were)\s+(?:not\s+(?:read|known|checked|reported|confirmed|visible|"
    r"available)|unknown|unavailable|unreadable)|(?:could|can)\s*not\s+be\s+\w+|unknown)\b",
    re.IGNORECASE,
)
_LEADING_WORDS = re.compile(r"^(?:so|and|but|then|while|its|their|his|her|the|a|an|all|every|any|some)\s+",
                            re.IGNORECASE)
_PHRASE_END = re.compile(r"\s(?:of|on|for|in|from|with|to|at)\s")
_WORD = re.compile(r"[a-z0-9][a-z0-9/_.-]*")
_NOT_A_SUBJECT = {"it", "they", "this", "that", "these", "those", "which", "what", "there"}
# Words that say a thing is not known; a clause with one does not claim it.
_HEDGE = re.compile(
    r"\b(?:unknown|unavailable|unreadable|unclear|unable|cannot|can't|couldn't|"
    r"(?:can|could|does|do|did|is|are|was|were)\s*not\s+(?:be\s+)?(?:read|known|checked|confirmed|verified|"
    r"visible|available|reported|seen|determined|shown|show|tell|say|know)|not\s+known|no\s+visibility)\b",
    re.IGNORECASE,
)


def _head(subject: str) -> str | None:
    words = _WORD.findall(_PHRASE_END.split(f" {subject.lower()} ", 1)[0])
    return words[-1] if words and words[-1] not in _NOT_A_SUBJECT else None


def _same_word(word: str, head: str) -> bool:
    def stem(text: str) -> str:
        text = text.removesuffix("'s")
        return text[:-1] if len(text) > 4 and text.endswith("s") and not text.endswith("ss") else text

    word, head = stem(word), stem(head)
    if min(len(word), len(head)) < 4:
        return word == head
    if word.startswith(head) or head.startswith(word):
        return True
    # One stem in two forms ("expires" and "expiry", "covered" and "coverage").
    shared = len(os.path.commonprefix([word, head]))
    return shared >= 5 and shared >= 0.6 * min(len(word), len(head))


def unknown_subjects(reading: str) -> list[str]:
    """The subjects a reading says it could not read or does not know, in order."""
    subjects: list[str] = []
    for match in _UNREADABLE_LIST.finditer(reading or ""):
        subjects += [item.strip() for item in re.split(r",|\band\b", match.group(1)) if item.strip()]
    for clause in re.split(r"[.;:,()\n]+", reading or ""):
        clause = clause.strip()
        while _LEADING_WORDS.match(clause):
            clause = _LEADING_WORDS.sub("", clause, count=1)
        match = _UNKNOWN_CLAUSE.match(clause)
        if match and _head(match.group("subject")):
            subjects.append(match.group("subject").strip())
    return list(dict.fromkeys(subjects))


def states_an_unknown(text: str, subjects: list[str]) -> bool:
    """Whether a clause of the text names one of the subjects without saying it is not known.

    A word check, so it fails safe: a false alarm only replaces an answer with
    the exact reading. It cannot catch a guess that shares no word with the
    subject ("all packages are up to date" for pending updates).
    """
    heads = [head for head in (_head(subject) for subject in subjects) if head]
    if not heads:
        return False
    for clause in re.split(r"[.;!?,\n]+|\bbut\b", text or ""):
        if _HEDGE.search(clause):
            continue
        words = _WORD.findall(clause.lower())
        if any(_same_word(word, head) for word in words for head in heads):
            return True
    return False


def exact_readings(results: list[AgentResult]) -> str | None:
    """Every result's complete reading, when each result is a successful read that declares one."""
    readings = []
    for result in results:
        tool = get_tool(result.proposed_tool) if result.proposed_tool else None
        answer = (result.answer or "").strip()
        if result.status != "success" or (tool or {}).get("synthesis_evidence") != "answer" or not answer:
            return None
        readings.append(answer)
    return "\n\n".join(readings) if readings else None


class PrimaryAssistant:
    """
    Main conversational model.

    It serves two purposes:

    1. direct responses when no specialist is needed;
    2. synthesis of successful trusted specialist results.

    It does not own model weights or runtime resources.
    """

    def __init__(
        self,
        inference: InferenceEngine,
        model_key: str,

        response_max_new_tokens: (
            int | None
        ) = None,

        synthesis_max_new_tokens: (
            int | None
        ) = None,

        capabilities: Callable[[], Any] | None = None,
    ) -> None:

        self.inference = (
            inference
        )

        self.model_key = (
            model_key
        )

        # Reads the current setup rows (as /v1/integrations returns them), so a
        # direct answer about what Janus can do follows the live registry.
        self.capabilities = capabilities

        settings = (
            get_settings()
        )

        self.response_max_new_tokens = (
            response_max_new_tokens
            if response_max_new_tokens
            is not None
            else (
                settings
                .primary_response_max_new_tokens
            )
        )

        self.synthesis_max_new_tokens = (
            synthesis_max_new_tokens
            if synthesis_max_new_tokens
            is not None
            else (
                settings
                .primary_synthesis_max_new_tokens
            )
        )

        if (
            not isinstance(
                self.response_max_new_tokens,
                int,
            )
            or self.response_max_new_tokens
            < 1
        ):

            raise ValueError(
                "Primary response max_new_tokens "
                "must be a positive integer."
            )

        if (
            not isinstance(
                self.synthesis_max_new_tokens,
                int,
            )
            or self.synthesis_max_new_tokens
            < 1
        ):

            raise ValueError(
                "Primary synthesis max_new_tokens "
                "must be a positive integer."
            )

        self.system_prompt = (
            load_prompt(
                "primary_assistant.txt"
            )
        )

    async def respond(
        self,
        user_request: str,
        *,
        context: list[dict[str, Any]] | None = None,
    ) -> str:
        # A provider that reaches the output cap fails the job as truncated,
        # so the model is told the limit, as in synthesis.
        max_new_tokens = answer_token_limit(InferenceStage.PRIMARY_RESPONSE, self.response_max_new_tokens)
        word_limit = answer_word_limit(max_new_tokens)

        messages = [
            {
                "role":
                    "system",

                "content":
                    self.system_prompt
                    + f"\n\nAnswer in at most {word_limit} words.",
            },

            *conversation_messages(
                context,
                max_turns=24,
            ),

            *prior_result_messages(
                context,
                max_turns=24,
            ),

            *self._capability_messages(),

            {
                "role":
                    "user",

                "content":
                    user_request,
            },
        ]

        response = (
            await self.inference.generate(
                stage=InferenceStage.PRIMARY_RESPONSE,
                model_key=(
                    self.model_key
                ),
                messages=messages,
                max_new_tokens=max_new_tokens,
                priority=(
                    InferencePriority
                    .PRIMARY_RESPONSE
                ),
            )
        )

        return response.strip()

    def _capability_messages(self) -> list[dict[str, str]]:
        """Current capabilities for a direct answer; nothing when setup cannot be read."""
        if self.capabilities is None:
            return []
        try:
            return capability_messages(self.capabilities())
        except Exception:
            return []

    async def _summarize_figures(
        self,
        user_request: str,
        reading: str,
        *,
        context: list[dict[str, Any]] | None = None,
    ) -> str:
        """A figure-free summary, then the exact reading; the reading alone when no summary passes."""
        max_new_tokens = min(
            answer_token_limit(InferenceStage.SYNTHESIS, self.synthesis_max_new_tokens), SUMMARY_MAX_NEW_TOKENS,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            *conversation_messages(context, max_turns=24),
            {
                "role": "user",
                "content": (
                    "ORIGINAL USER REQUEST:\n"
                    f"{user_request}"
                    "\n\n"
                    "TOOL READING (shown to the user exactly as written, right after your sentences):\n"
                    f"{reading}"
                    "\n\n"
                    "Write one or two sentences that answer the request in words. Use no numbers: no digits, "
                    "no number words and no names or values that contain digits; the reading gives every "
                    "figure. State nothing the reading does not state. If the reading says something is not "
                    "read or unknown, say it is not known; never guess it."
                ),
            },
        ]
        try:
            response = await self.inference.generate(
                stage=InferenceStage.SYNTHESIS,
                model_key=self.model_key,
                messages=messages,
                max_new_tokens=max_new_tokens,
                priority=InferencePriority.PRIMARY_RESPONSE,
            )
        except Exception as exc:
            print(f"[HUB] Figure-free summary failed; the exact reading stands error_type={type(exc).__name__}")
            return reading
        summary = plain_summary(response)
        if summary is None:
            print("[HUB] Figure-free summary not used (a figure, too long or more than two sentences); "
                  "the exact reading stands")
            return reading
        if states_an_unknown(summary, unknown_subjects(reading)):
            print("[HUB] Figure-free summary not used (it states what the reading does not know); "
                  "the exact reading stands")
            return reading
        return f"{summary}\n\n{reading}"

    async def synthesize(
        self,
        user_request: str,
        results: list[
            AgentResult
        ],
        *,
        context: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Produce the final Main-LLM answer from successful
        specialist results.

        Structured results remain authoritative. The Main model
        is responsible only for user-facing synthesis.
        """

        reading = complete_reading(results)
        if reading is not None:
            return reading

        reading = reading_with_many_figures(results)
        if reading is not None:
            return await self._summarize_figures(user_request, reading, context=context)

        specialist_payload = [
            synthesis_evidence(
                result
            )

            for result in results
        ]

        max_new_tokens = answer_token_limit(InferenceStage.SYNTHESIS, self.synthesis_max_new_tokens)
        word_limit = answer_word_limit(max_new_tokens)

        messages = [
            {
                "role":
                    "system",

                "content":
                    self.system_prompt,
            },

            *conversation_messages(
                context,
                max_turns=24,
            ),

            *prior_result_messages(
                context,
                max_turns=24,
            ),

            {
                "role":
                    "user",

                "content": (
                    "ORIGINAL USER REQUEST:\n"
                    f"{user_request}"
                    "\n\n"
                    "TRUSTED SPECIALIST RESULTS:\n"
                    f"{json.dumps(
                        specialist_payload,
                        separators=(",", ":"),
                        default=str,
                    )}"
                    "\n\n"
                    "Synthesize the final answer "
                    "for the user using only these "
                    "specialist results for system "
                    "or external-state claims. "
                    f"Use at most {word_limit} words "
                    "and no tables. For long results "
                    "give counts and at most 10 of "
                    "the most relevant items."
                ),
            },
        ]

        response = (
            await self.inference.generate(
                stage=InferenceStage.SYNTHESIS,
                model_key=(
                    self.model_key
                ),
                messages=(
                    messages
                ),
                max_new_tokens=max_new_tokens,
                priority=(
                    InferencePriority
                    .PRIMARY_RESPONSE
                ),
            )
        )

        answer = response.strip()
        readings = exact_readings(results)
        if readings is not None and states_an_unknown(answer, unknown_subjects(readings)):
            print("[HUB] Synthesized answer not used (it states what a reading does not know); "
                  "the exact readings stand")
            return readings
        return answer
