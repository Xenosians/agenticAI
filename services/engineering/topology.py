"""Desired infrastructure topologies compared with an observation (FR-093, V19-09).

A topology is the owner's desired graph. An observation is what the fixed read
adapters saw. The comparison is advisory "simulated" state: drift and dependency
reach are reported, and nothing is ever applied. Only properties a topology
declares are compared, so volatile observed facts never count as drift.
"""
from __future__ import annotations

from collections import deque
import math
import re

NODE_ID = re.compile(r"[a-z][a-z0-9_]{0,31}:[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}")
KIND = re.compile(r"[a-z][a-z0-9_]{0,31}")
PROPERTY = re.compile(r"[a-z][a-z0-9_]{0,63}")
RELATIONS = ("depends_on", "hosted_on", "connected_to", "associated_with")
PRIVATE_NAME = re.compile(r"pass(word|wd)?|secret|token|credential|api_?key|private_?key|cookie|authorization")
# Credential and personal-data shapes, shared with Phoenix's topology validation.
# Ordinary words such as "Basic switch" or "Token ring" are not credentials.
PRIVATE_TEXT = re.compile(
    r"\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{16,}"
    r"|\b(?:password|passwd|secret|api[_-]?key|token|client[_-]?secret)\s*[:=]\s*[^\s,;]+"
    r"|https?://[^/\s:@]+:[^@\s/]+@"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY"
    r"|\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b",
    re.IGNORECASE,
)
MAX_NODES, MAX_EDGES, MAX_PROPERTIES = 300, 1200, 40
MAX_LABEL, MAX_TEXT = 120, 200
SAFE_INTEGER = 2 ** 53
# Each observation source covers the node kinds whose identity prefix it reads.
SOURCE_OF_PREFIX = {"container": "containers", "endpoint": "endpoints", "router": "router", "interface": "router"}


class TopologyError(ValueError):
    """A refused topology; the code is public, the offending value never is."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _refuse(message: str):
    raise TopologyError("invalid_topology", message)


def _text(value, limit: int, field: str) -> str:
    if not isinstance(value, str) or len(value) > limit or any(ch in value for ch in "\r\n\t\x00"):
        _refuse(f"{field} must be single-line text of at most {limit} characters.")
    if PRIVATE_TEXT.search(value):
        raise TopologyError("private_topology_content", "Topologies cannot hold credentials or personal data.")
    return value


def _value(value, field: str):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) > SAFE_INTEGER:
            _refuse(f"{field} is outside the exact integer range.")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _refuse(f"{field} must be a finite number.")
        return value
    if isinstance(value, str):
        return _text(value, MAX_TEXT, field)
    _refuse(f"{field} must be text, a number, true/false or null.")


def _properties(value) -> dict:
    if not isinstance(value, dict) or len(value) > MAX_PROPERTIES:
        _refuse(f"Properties must be an object with at most {MAX_PROPERTIES} entries.")
    result = {}
    for name, item in value.items():
        if not isinstance(name, str) or not PROPERTY.fullmatch(name):
            _refuse("Property names are lowercase letters, digits and underscores.")
        if PRIVATE_NAME.search(name):
            raise TopologyError("private_topology_content", "Topologies cannot hold credentials or personal data.")
        result[name] = _value(item, "A property value")
    return result


def _exact_keys(value, keys: set[str], what: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        _refuse(f"Each {what} has exactly the fields {', '.join(sorted(keys))}.")
    return value


def parse_topology(document) -> dict:
    """Validate a desired topology and return its canonical form."""
    document = _exact_keys(document, {"nodes", "edges"}, "topology")
    nodes, edges = document["nodes"], document["edges"]
    if not isinstance(nodes, list) or not isinstance(edges, list) or len(nodes) > MAX_NODES or len(edges) > MAX_EDGES:
        _refuse(f"A topology has at most {MAX_NODES} nodes and {MAX_EDGES} edges.")
    parsed_nodes, seen = [], set()
    for raw in nodes:
        node = _exact_keys(raw, {"id", "kind", "label", "properties"}, "node")
        if not isinstance(node["id"], str) or not NODE_ID.fullmatch(node["id"]) or node["id"] in seen:
            _refuse("Node identities are unique 'type:name' values.")
        if not isinstance(node["kind"], str) or not KIND.fullmatch(node["kind"]):
            _refuse("Node kinds are lowercase letters, digits and underscores.")
        seen.add(node["id"])
        parsed_nodes.append({"id": node["id"], "kind": node["kind"], "label": _text(node["label"], MAX_LABEL, "A label"),
                             "properties": _properties(node["properties"])})
    parsed_edges, links = [], set()
    for raw in edges:
        edge = _exact_keys(raw, {"from", "to", "relation"}, "edge")
        source, target, relation = edge["from"], edge["to"], edge["relation"]
        if source not in seen or target not in seen or source == target:
            _refuse("Edges join two different nodes of the topology.")
        if relation not in RELATIONS or (source, target, relation) in links:
            _refuse("Edges use a known relation and appear once.")
        links.add((source, target, relation))
        parsed_edges.append({"from": source, "to": target, "relation": relation})
    return {"nodes": parsed_nodes, "edges": parsed_edges}


def _node_source(node_id: str) -> str | None:
    return SOURCE_OF_PREFIX.get(node_id.split(":", 1)[0])


def _differences(expected: dict, observed: dict) -> list[dict]:
    rows = []
    for name in sorted(expected):
        present = name in observed
        actual = observed.get(name)
        # Strict JSON identity: true is not 1, and 1 is not 1.0.
        if not present or type(actual) is not type(expected[name]) or actual != expected[name]:
            rows.append({"property": name, "expected": expected[name], "observed": actual, "observed_present": present})
    return rows


def _status(node: dict, observed: dict, absent: set, unread: set, sources: dict) -> dict:
    row = {"id": node["id"], "status": "unknown", "reason": None, "differences": []}
    source = _node_source(node["id"])
    if source is None:
        row["reason"] = "no_source"
        return row
    state = sources.get(source, {"state": "unavailable"})["state"]
    if node["id"] in observed:
        row["differences"] = _differences(node["properties"], observed[node["id"]]["properties"])
        row["status"] = "drifted" if row["differences"] else "matches"
        return row
    if node["id"] in absent:
        row["status"] = "missing"
        return row
    if state == "not_configured":
        row["reason"] = "source_not_configured"
    elif state != "observed":
        row["reason"] = "source_unavailable"
    elif node["id"] in unread:
        row["reason"] = "not_read"
    else:
        row["reason"] = "not_in_source_scope"
    return row


def _reach(edges: list[dict], seeds: set[str]) -> list[str]:
    affected: dict[str, set[str]] = {}
    for edge in edges:
        if edge["relation"] in ("depends_on", "hosted_on"):
            # A depends on / is hosted on B: a change to B may affect A.
            affected.setdefault(edge["to"], set()).add(edge["from"])
        elif edge["relation"] == "connected_to":
            affected.setdefault(edge["to"], set()).add(edge["from"])
            affected.setdefault(edge["from"], set()).add(edge["to"])
    reached, queue = set(seeds), deque(sorted(seeds))
    while queue:
        for dependent in sorted(affected.get(queue.popleft(), ())):
            if dependent not in reached:
                reached.add(dependent)
                queue.append(dependent)
    return sorted(reached - seeds)


def compare(topology: dict, observation: dict, *, topology_id: str, version: int) -> dict:
    """Advisory drift of one saved topology version against one observation."""
    observed = {node["id"]: node for node in observation["nodes"]}
    absent, unread = set(observation["absent"]), set(observation["unread"])
    sources = {row["id"]: row for row in observation["sources"]}
    rows = []
    for node in topology["nodes"]:
        row = _status(node, observed, absent, unread, sources)
        if row["reason"] is None:
            del row["reason"]
        rows.append(row)
    changed = {row["id"] for row in rows if row["status"] in ("drifted", "missing")}
    designed = {node["id"] for node in topology["nodes"]}
    counts = {status: sum(1 for row in rows if row["status"] == status)
              for status in ("matches", "drifted", "missing", "unknown")}
    return {
        "schema": "topology-drift.v1", "state_kind": "simulated",
        "topology_id": topology_id, "topology_version": version,
        "observed_at": observation["observed_at"], "observation_sha256": observation["observation_sha256"],
        "sources": observation["sources"], "nodes": rows, "counts": counts,
        "potentially_affected": _reach(topology["edges"], changed),
        "not_in_design": sorted(set(observed) - designed),
        "execution_enabled": False, "apply_to_live": False,
        "notes": ["Only properties the topology declares are compared.",
                  "Potentially affected means graph reachability, not a measured outage.",
                  "A drift report never changes a device; live changes use governed, approved tools."],
    }
