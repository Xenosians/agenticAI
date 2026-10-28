"""KiCad netlist inspection (FR-093 PCB/EDA design review).

A netlist is a design artifact, so it is desired state. Inspection parses text
only and reports deterministic connectivity findings; it never opens files,
runs EDA tools or drives fabrication or programming hardware.
"""
from __future__ import annotations

import hashlib
import re

MAX_BYTES = 2_000_000
MAX_DEPTH = 32
MAX_COMPONENTS = 5000
MAX_NETS = 5000
MAX_CONNECTIONS = 50_000
MAX_LISTED_NODES = 50
MAX_FINDINGS = 500
TOKEN = re.compile(r'\s*(?:(\()|(\))|"([^"\\]*(?:\\.[^"\\]*)*)"|([^\s()"]+))', re.S)
ESCAPE = re.compile(r"\\(.)", re.S)
ESCAPES = {"n": "\n", "t": "\t", "r": "\r"}
REFERENCE = re.compile(r"[^\s()\"]{1,64}")
PIN = re.compile(r"[^\s()\"]{1,32}")
SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
MESSAGES = {
    "duplicate_reference": "More than one component uses this reference.",
    "unannotated_reference": "Annotate the schematic before layout.",
    "missing_footprint": "This component has no footprint, so it cannot be placed on the board.",
    "unknown_component": "A net connects to a reference that is not in the component list.",
    "pin_on_multiple_nets": "This pin is on more than one net, which shorts those nets together.",
    "single_connection_net": "This net reaches only one pin, so it connects nothing.",
    "unconnected_pin": "This pin is left unconnected; confirm that is intended.",
    "no_connections": "No pin of this component is on a net; expected only for mechanical parts.",
}
SEVERITY = {"duplicate_reference": "error", "unannotated_reference": "error", "missing_footprint": "error",
            "unknown_component": "error", "pin_on_multiple_nets": "error", "single_connection_net": "warning",
            "unconnected_pin": "info", "no_connections": "info"}


class NetlistError(ValueError):
    """A refused netlist; the message describes the format, never the content."""

    def __init__(self, message: str):
        super().__init__(message)
        self.code = "invalid_netlist"


def _parse(text: str) -> list:
    stack, position = [[]], 0
    while True:
        match = TOKEN.match(text, position)
        if match is None:
            if text[position:].strip():
                raise NetlistError("The netlist has an unterminated string or an invalid token.")
            break
        position = match.end()
        opened, closed, quoted, atom = match.groups()
        if opened:
            if len(stack) > MAX_DEPTH:
                raise NetlistError("The netlist nests too deeply.")
            stack.append([])
        elif closed:
            if len(stack) == 1:
                raise NetlistError("The netlist has an unmatched closing parenthesis.")
            finished = stack.pop()
            stack[-1].append(finished)
        elif quoted is not None:
            stack[-1].append(ESCAPE.sub(lambda m: ESCAPES.get(m.group(1), m.group(1)), quoted))
        else:
            stack[-1].append(atom)
    if len(stack) != 1:
        raise NetlistError("The netlist has an unclosed parenthesis.")
    items = stack[0]
    if len(items) != 1 or not isinstance(items[0], list) or not items[0] or items[0][0] != "export":
        raise NetlistError("A KiCad netlist is one (export ...) expression.")
    return items[0]


def _sections(expression: list, name: str) -> list[list]:
    return [item for item in expression[1:] if isinstance(item, list) and item and item[0] == name]


def _atom(expression: list, name: str) -> str | None:
    found = _sections(expression, name)
    if not found:
        return None
    values = found[0][1:]
    if len(values) != 1 or not isinstance(values[0], str):
        raise NetlistError(f"The {name} field holds exactly one value.")
    return values[0]


def _required(expression: list, name: str, pattern: re.Pattern, what: str) -> str:
    value = _atom(expression, name)
    if value is None or not pattern.fullmatch(value):
        raise NetlistError(f"Each entry needs a {name}; {what} are short values without spaces or parentheses.")
    return value


def _text(value: str | None, limit: int) -> str | None:
    return None if value is None else " ".join(value.split())[:limit]


def _natural(value: str) -> list:
    return [(0, int(part), "") if part.isdigit() else (1, 0, part) for part in re.split(r"(\d+)", value)]


def _components(export: list) -> list[dict]:
    rows = [item for section in _sections(export, "components") for item in _sections(section, "comp")]
    if len(rows) > MAX_COMPONENTS:
        raise NetlistError(f"A netlist has at most {MAX_COMPONENTS} components.")
    components = []
    for comp in rows:
        library = _sections(comp, "libsource")
        lib = _atom(library[0], "lib") if library else None
        part = _atom(library[0], "part") if library else None
        components.append({
            "ref": _required(comp, "ref", REFERENCE, "component references"),
            "value": _text(_atom(comp, "value"), 120),
            "footprint": _text(_atom(comp, "footprint"), 200) or None,
            "part": _text(f"{lib}:{part}" if lib and part else part, 200) if part else None,
        })
    return sorted(components, key=lambda row: _natural(row["ref"]))


def _nets(export: list) -> tuple[list[dict], dict, int]:
    rows = [item for section in _sections(export, "nets") for item in _sections(section, "net")]
    if len(rows) > MAX_NETS:
        raise NetlistError(f"A netlist has at most {MAX_NETS} nets.")
    nets, pin_nets, connections = [], {}, 0
    for index, net in enumerate(rows):
        nodes = [(_required(node, "ref", REFERENCE, "component references"),
                  _required(node, "pin", PIN, "pin numbers"))
                 for node in _sections(net, "node")]
        connections += len(nodes)
        if connections > MAX_CONNECTIONS:
            raise NetlistError(f"A netlist has at most {MAX_CONNECTIONS} connections.")
        for node in nodes:
            pin_nets.setdefault(node, set()).add(index)
        labels = sorted({f"{reference}.{pin}" for reference, pin in nodes}, key=_natural)
        nets.append({"code": _text(_atom(net, "code"), 32), "name": _text(_atom(net, "name"), 200) or "",
                     "connections": len(nodes), "nodes": labels[:MAX_LISTED_NODES]})
    return nets, pin_nets, connections


def _findings(components: list[dict], nets: list[dict], pin_nets: dict) -> list[dict]:
    found: list[tuple[str, str]] = []
    references: dict[str, int] = {}
    for row in components:
        references[row["ref"]] = references.get(row["ref"], 0) + 1
    connected = {reference for reference, _pin in pin_nets}
    for reference, count in references.items():
        if count > 1:
            found.append(("duplicate_reference", reference))
        if reference.endswith("?"):
            found.append(("unannotated_reference", reference))
        if reference not in connected:
            found.append(("no_connections", reference))
    found += [("missing_footprint", reference) for reference in {row["ref"] for row in components if not row["footprint"]}]
    found += [("unknown_component", reference) for reference in connected - set(references)]
    found += [("pin_on_multiple_nets", f"{reference}.{pin}") for (reference, pin), indexes in pin_nets.items()
              if len(indexes) > 1]
    for net in nets:
        if net["name"].startswith("unconnected-("):
            found.append(("unconnected_pin", net["name"]))
        elif net["connections"] == 1:
            found.append(("single_connection_net", net["name"]))
    rows = [{"severity": SEVERITY[code], "code": code, "subject": subject, "message": MESSAGES[code]}
            for code, subject in found]
    return sorted(rows, key=lambda row: (SEVERITY_ORDER[row["severity"]], row["code"], _natural(row["subject"])))


def inspect_netlist(text: str) -> dict:
    """Parse one KiCad netlist export and report its parts, nets and design findings."""
    if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > MAX_BYTES:
        raise NetlistError("A netlist is nonempty text of at most 2 MB.")
    export = _parse(text)
    design = _sections(export, "design")
    components = _components(export)
    nets, pin_nets, connections = _nets(export)
    pins: dict[str, int] = {}
    for reference, _pin in pin_nets:
        pins[reference] = pins.get(reference, 0) + 1
    for row in components:
        row["pins"] = pins.get(row["ref"], 0)
    findings = _findings(components, nets, pin_nets)
    return {
        "schema": "pcb-netlist-inspection.v1", "state_kind": "desired", "actuation": "none",
        "format": "kicad", "tool": _text(_atom(design[0], "tool"), 120) if design else None,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "summary": {"components": len(components), "nets": len(nets), "connections": connections,
                    "errors": sum(1 for row in findings if row["severity"] == "error"),
                    "warnings": sum(1 for row in findings if row["severity"] == "warning")},
        "components": components, "nets": nets,
        "findings": findings[:MAX_FINDINGS], "findings_truncated": len(findings) > MAX_FINDINGS,
    }
