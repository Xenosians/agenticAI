"""Local, non-authoritative catalog retrieval for compact Hub disclosure.

Candidates are search hints, never a semantic intent or an execution allowlist.
The router, SemanticGuard and ToolGateway must continue using the full registry.
Only trusted registry metadata is disclosed; selector-provided descriptors are
neither accepted nor copied into a model prompt.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Protocol

from services.retrieval.fts import FtsRecord, search_fts
from services.extensions.budget import estimate_text_tokens
from subagents.core.definitions.registry import AgentRegistry
from subagents.core.orchestration.intent_contract import (
    trusted_condition_fields,
    trusted_derived_arguments,
    trusted_grounded_argument_descriptions,
    trusted_grounded_arguments,
    trusted_policy_owns_preconditions,
    trusted_required_arguments,
    trusted_requires_approval,
    trusted_tool_effect,
)
from tools.registry import get_tool
from subagents.core.orchestration.router_catalog import render_router_catalog


@dataclass(frozen=True)
class Candidate:
    agent_name: str
    capability_names: tuple[str, ...]


@dataclass(frozen=True)
class CandidateSet:
    candidates: tuple[Candidate, ...] = ()
    confidence: float = 0.0
    widening_count: int = 0
    fallback_required: bool = False
    reason: str = ""

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def capability_count(self) -> int:
        return len({name for item in self.candidates for name in item.capability_names})


class CandidateSelector(Protocol):
    def select(self, user_request: str, registry: AgentRegistry) -> CandidateSet:
        """Return untrusted selection metadata, never an executable route."""
        ...


class CandidateValidationError(ValueError):
    """A selector returned invalid identities, ownership or accounting fields."""


def validate_candidate_set(selection: CandidateSet, registry: AgentRegistry) -> CandidateSet:
    """Recheck every identity against the current registry before disclosure.

    Reject the whole result on any invalid item; silently retaining the valid
    subset could turn a malicious or buggy selector into capability substitution.
    """
    if not isinstance(selection, CandidateSet) or not isinstance(selection.candidates, tuple):
        raise CandidateValidationError("Invalid candidate set")
    if (type(selection.confidence) not in (int, float)
            or not math.isfinite(selection.confidence)
            or not 0 <= selection.confidence <= 1):
        raise CandidateValidationError("Invalid candidate confidence")
    if type(selection.widening_count) is not int or selection.widening_count < 0:
        raise CandidateValidationError("Invalid candidate widening count")
    if type(selection.fallback_required) is not bool or not isinstance(selection.reason, str):
        raise CandidateValidationError("Invalid candidate fallback metadata")
    seen_agents = set()
    for candidate in selection.candidates:
        if not isinstance(candidate, Candidate) or not isinstance(candidate.agent_name, str):
            raise CandidateValidationError("Invalid candidate identity")
        if not registry.exists(candidate.agent_name):
            raise CandidateValidationError("Unknown candidate specialist")
        if candidate.agent_name in seen_agents:
            raise CandidateValidationError("Duplicate candidate specialist")
        seen_agents.add(candidate.agent_name)
        if not isinstance(candidate.capability_names, tuple) or not candidate.capability_names:
            raise CandidateValidationError("Invalid candidate capabilities")
        owned = set(registry.get(candidate.agent_name).tools)
        seen_tools = set()
        for name in candidate.capability_names:
            if not isinstance(name, str) or not name or name not in owned or get_tool(name) is None:
                raise CandidateValidationError("Unknown or unowned candidate capability")
            if name in seen_tools:
                raise CandidateValidationError("Duplicate candidate capability")
            seen_tools.add(name)
    return selection


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1].rsplit(" ", 1)[0] + "…"


def _capability_descriptor(name: str) -> dict:
    tool = get_tool(name)
    if tool is None:
        raise CandidateValidationError("Unknown candidate capability")
    description = tool.get("description")
    if not isinstance(description, str) or not description.strip():
        raise CandidateValidationError("Invalid trusted capability description")
    args = {"tool_name": name, "tool": tool}
    metadata = {
        "effect": trusted_tool_effect(tool),
        "grounded_arguments": trusted_grounded_arguments(**args),
        "required_arguments": trusted_required_arguments(**args),
        "derived_arguments": trusted_derived_arguments(**args),
        "condition_fields": trusted_condition_fields(**args),
        "requires_approval": trusted_requires_approval(**args),
        "policy_owns_preconditions": trusted_policy_owns_preconditions(**args),
    }
    # Parameter descriptions preserve binding meaning, without argument values,
    # examples, full schemas, runtime resolvers or implementation callables.
    descriptions = trusted_grounded_argument_descriptions(**args)
    if descriptions:
        metadata["grounded_argument_descriptions"] = {
            key: _short(value, 120) for key, value in descriptions.items()
        }
    descriptor = {
        "name": name,
        "description": _short(description, 200),
        "risk": tool["risk"],
        "intent_metadata": metadata,
    }
    for key in ("resource_type", "operation_kind"):
        if isinstance(tool.get(key), str) and tool[key]:
            descriptor[key] = tool[key]
    return descriptor


def build_candidate_agent_specs(selection: CandidateSet, registry: AgentRegistry) -> list[dict]:
    """Rebuild a compact, trusted subset in the existing router catalog shape."""
    validate_candidate_set(selection, registry)
    return [
        {
            "name": candidate.agent_name,
            "description": _short(registry.get(candidate.agent_name).description, 240),
            "capabilities": [_capability_descriptor(name) for name in candidate.capability_names],
        }
        for candidate in selection.candidates
    ]


# Grammatical stop words only: no domain names, operation synonyms or routing
# table. All domain/operation evidence comes from the current trusted registry.
_STOP_WORDS = frozenset("a an and are as at be by can could do does for from how i if in is it its me my of on or please the then this to was what when where which with would you your".split())


def _terms(text: str) -> set[str]:
    words = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
    result = set()
    for word in words:
        if len(word) < 2 or word.isdecimal() or word in _STOP_WORDS:
            continue
        # A generic plural normalization shared by query and catalog text.
        if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us")):
            word = word[:-1]
        result.add(word)
    return result


class DeterministicCandidateSelector:
    """FTS retrieval with conservative widening and explicit local fallback.

    FTS and token coverage rank descriptions, not user authority. Shared tools
    retain every registered owner. Low evidence, an unresolved boundary tie or
    an oversized query requests broad local routing instead of guessing.
    """

    def __init__(self, *, top_k: int = 6, widen_k: int = 12, max_query_terms: int = 64,
                 max_packet_tokens: int = 2500) -> None:
        if any(type(value) is not int or value < 1 for value in (top_k, widen_k, max_query_terms, max_packet_tokens)):
            raise ValueError("Candidate limits must be positive integers")
        if widen_k < top_k:
            raise ValueError("Widened candidate limit must not be smaller than top-k")
        self.top_k = top_k
        self.widen_k = widen_k
        self.max_query_terms = max_query_terms
        self.max_packet_tokens = max_packet_tokens

    def select(self, user_request: str, registry: AgentRegistry) -> CandidateSet:
        if not isinstance(user_request, str):
            raise ValueError("Candidate request must be text")
        query = _terms(user_request)
        if not query:
            return CandidateSet(fallback_required=True, reason="no_query_evidence")
        if len(query) > self.max_query_terms:
            return CandidateSet(fallback_required=True, reason="query_limit")

        owners = defaultdict(list)
        agents = sorted(registry.list_agents(), key=lambda agent: agent.name)
        for agent in agents:
            for name in sorted(set(agent.tools)):
                if get_tool(name) is None:
                    raise CandidateValidationError("Registry references unknown capability")
                owners[name].append(agent)
        if not owners:
            return CandidateSet(fallback_required=True, reason="empty_registry")

        records, terms, identity_terms, capability_terms = [], {}, {}, {}
        for name, agent_owners in sorted(owners.items()):
            tool = get_tool(name)
            identity = " ".join((name, tool.get("resource_type", ""), tool.get("operation_kind", ""),
                                 tool.get("permission", "")))
            title = identity + " " + " ".join(agent.name for agent in agent_owners)
            content = " ".join((tool.get("description", ""), *(agent.description for agent in agent_owners)))
            terms[name] = _terms(title + " " + content)
            identity_terms[name] = _terms(identity)
            capability_terms[name] = _terms(identity + " " + tool.get("description", ""))
            records.append(FtsRecord(name, " ".join(sorted(_terms(content))), title=" ".join(sorted(_terms(title)))))
        hits = search_fts(records, " ".join(sorted(query)), limit=len(records))
        if not hits.identifiers:
            return CandidateSet(fallback_required=True, reason="no_matches")

        frequencies = Counter(term for record_terms in terms.values() for term in record_terms)
        weights = {term: math.log1p(len(records) / frequencies[term]) for term in query if frequencies[term]}
        known = set(weights)
        # Capability identities describe the operations and resources directly.
        # When those supply evidence, rare incidental prose (for example a
        # display verb occurring in one description) must not dominate it.
        identity_evidence = known & set().union(*identity_terms.values())
        if len(identity_evidence) >= 2:
            weights = {term: value if term in identity_evidence else value * 0.25
                       for term, value in weights.items()}
        # A lone generic word appearing in most records is inadequate evidence.
        informative = {term for term in known if frequencies[term] <= max(1, len(records) // 2)}
        total_weight = sum(weights.values()) or 1.0
        scores = {
            name: sum(
                weights[term] * (2.0 if term in identity_terms[name]
                                 else 1.0 if term in capability_terms[name] else 0.25)
                for term in known & terms[name]
            ) / (2 * total_weight)
            for name in hits.identifiers
        }
        # FTS provides the relevance ordering; deterministic coverage breaks
        # domain/operation omissions before FTS relevance and stable identity.
        positions = {name: i for i, name in enumerate(hits.identifiers)}
        ranked = sorted(hits.identifiers, key=lambda name: (-scores[name], positions[name], name))
        best = ranked[0]
        # Incidental matches in broad owner descriptions must not force every
        # specialist into a tiny packet. A weak best match still falls back.
        ranked = [name for name in ranked if scores[name] >= scores[best] * 0.45]
        evidence = (informative | identity_evidence) & capability_terms[best]
        confidence_terms = identity_evidence if len(identity_evidence) >= 2 else known
        coverage = (sum(weights[term] for term in confidence_terms & terms[best])
                    / (sum(weights[term] for term in confidence_terms) or 1.0))
        confidence = round(coverage * min(1.0, len(evidence) / 2), 4)
        weak = confidence < 0.6

        def boundary_tied(limit: int) -> bool:
            if len(ranked) <= limit:
                return False
            left, right = ranked[limit - 1], ranked[limit]
            tied = (abs(scores[left] - scores[right]) < 1e-9
                    and query & identity_terms[left] == query & identity_terms[right])
            if not tied:
                return False

            def strictly_dominated(name: str) -> bool:
                identity = query & identity_terms[name]
                capability = query & capability_terms[name]
                return any(
                    identity <= query & identity_terms[earlier]
                    and capability <= query & capability_terms[earlier]
                    and (identity < query & identity_terms[earlier]
                         or capability < query & capability_terms[earlier])
                    for earlier in ranked[:limit - 1]
                )

            # A tail tie with strictly weaker query evidence than an earlier
            # capability does not make the leading candidates ambiguous. Truly
            # indistinguishable candidates still widen or require fallback.
            return not (strictly_dominated(left) and strictly_dominated(right))

        # Account for the whole request, not only the strongest matching task.
        # Every informative catalog term needs capability-level coverage; broad
        # specialist prose cannot cover a second operation on its own. Splitting
        # grammatical conjunctions also exposes a wholly unmatched later task.
        required_evidence = informative & confidence_terms
        # Exact code literals often supply the only operation evidence (for
        # example an executable name). An unknown literal must not disappear
        # while generic prose makes an unrelated packet appear complete.
        # This does not interpret the literal or resolve runtime argument values.
        literal_terms = set().union(*(_terms(value) for value in re.findall(r"`([^`\n]+)`", user_request)))
        unmatched_literal = bool(literal_terms - known)
        clauses = re.split(r"\b(?:and|then|or)\b|[;\n]", user_request, flags=re.IGNORECASE)
        clause_terms = [_terms(clause) for clause in clauses]
        # A conjunction can describe outputs of one capability rather than a
        # separate operation. Its trusted description is valid lexical support
        # even when no name/resource token appears in that output phrase. Keep
        # unfamiliar clause terms unresolved instead of silently discarding
        # them and letting a generic pair such as "run command" cover an
        # otherwise unsupported action.
        def request_coverage(selected: list[str]) -> tuple[bool, bool]:
            covered = set().union(*(capability_terms[name] for name in selected))
            missing_evidence = bool(required_evidence - covered)
            unmatched_clause = len(clauses) > 1 and any(
                clause and (bool(clause - known)
                            or len(clause & known & covered) < min(2, len(clause)))
                for clause in clause_terms
            )
            return missing_evidence, unmatched_clause

        # A strong first operation may put the next operation just outside the
        # initial packet. Use the existing bounded widening opportunity before
        # declaring that request uncovered; never discard the missing clause.
        initial_missing, initial_unmatched = request_coverage(ranked[:self.top_k])
        widened = weak or boundary_tied(self.top_k) or initial_missing or initial_unmatched
        limit = self.widen_k if widened else self.top_k
        unresolved = boundary_tied(limit)
        selected = ranked[:limit]
        missing_evidence, unmatched_clause = request_coverage(selected)
        by_agent = defaultdict(list)
        for name in selected:
            for agent in owners[name]:
                by_agent[agent.name].append(name)
        result = CandidateSet(
            candidates=tuple(Candidate(name, tuple(capabilities)) for name, capabilities in sorted(by_agent.items())),
            confidence=confidence,
            widening_count=int(widened),
            fallback_required=weak or unresolved or missing_evidence or unmatched_clause or unmatched_literal,
            reason=("weak_evidence" if weak else "ambiguous_boundary" if unresolved
                    else "uncovered_request" if missing_evidence or unmatched_clause or unmatched_literal
                    else "widened" if widened else "matched"),
        )
        result = validate_candidate_set(result, registry)
        if not result.fallback_required:
            packet = render_router_catalog(build_candidate_agent_specs(result, registry))
            if estimate_text_tokens(packet) > self.max_packet_tokens:
                return CandidateSet(result.candidates, result.confidence, result.widening_count,
                                    fallback_required=True, reason="packet_limit")
        return result
