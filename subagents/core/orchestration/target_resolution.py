"""Human context: find the guest a request means in live Proxmox inventory.

Owner decision, 26 November 2026 (Implementation_Direction_Idea.md): "Look it up in live inventory: one
clear match is proposed with its real ID and name for your approval; several matches, you pick from a
short list; none, it says so".

People name a guest the way they think of it ("the lab container", "the trainer", "janus-lab", "guest
100"), not as the exact node name and integer VMID a governed call needs. When the routing contract
found the capability but not that identity in the request, this module reads the account's own Proxmox
inventory and matches the request's words against each guest's real name, tags and VMID:

* exactly one guest matches: its node and VMID become the target, and a trusted line naming them is
  added to the text the gateway grounds against. Nothing else is relaxed: binding, SemanticGuard,
  ToolGateway, the policy freeze of type and name, the owner's approval and the typed name of a
  destructive change all run on those real values, exactly as if the user had typed them;
* several match: the owner is asked which one, from a short list with each guest's real ID and name;
* none match, or inventory cannot be read: Janus says so and asks; it never guesses.

The identity always comes from Proxmox's listing. A model never supplies or chooses it, and nothing else
(a snapshot name, a size, a template, a disk, where something new goes) is looked up here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

# Words that say what kind of thing or which action is meant, never which guest.
GENERIC_WORDS = frozenset("""
a an the this that these those it its my our your their his her please pls kindly can could would will
should do does did is are was were be been being am have has had i me we us you he she they them what
which who whom whose where when why how about for to of in on at by with from into onto over under up
down out off and or but if then than so as also just now again still all any every each some no not
never except without only
guest guests vm vms virtual machine machines container containers ct cts lxc lxcs qemu kvm instance
instances server servers box boxes host hosts node nodes system systems one ones thing things
start started starting restart restarted restarting reboot rebooted rebooting stop stopped stopping
shutdown shut halt kill force power turn boot status state check show see list read tell running run
alive dead delete deleted remove removed destroy destroyed wipe drop clone copy snapshot snapshots
backup backups restore rollback roll back resize grow cpu cpus core cores memory ram disk disks storage
tag tags option options config configuration settings proxmox pve cluster id vmid
""".split())

TOKEN = re.compile(r"[a-z0-9]+")
# A standalone number: not part of a word, a version, an address, a size suffix or a port.
NUMBER = re.compile(r"(?<![A-Za-z0-9.:/_-])([0-9]{1,9})(?![A-Za-z0-9.:/_-])")
MAX_LISTED = 8

# Everyday names for engines and roles a guest's name or tags carry.
WORD_FAMILIES = (
    frozenset({"postgresql", "postgres", "pg", "database", "db"}),
    frozenset({"mariadb", "mysql", "database", "db"}),
    frozenset({"control", "controller"}),
    frozenset({"trainer", "training"}),
)


@dataclass(frozen=True)
class GuestRecord:
    """One guest as the account's Proxmox lists it (templates left out)."""

    vmid: int
    node: str
    type: str
    name: str | None
    status: str
    tags: tuple[str, ...] = field(default=())

    @property
    def label(self) -> str:
        kind = "VM" if self.type == "qemu" else "LXC"
        return f"{self.name or 'unnamed'} ({kind} {self.vmid} on node {self.node}, {self.status})"


@dataclass(frozen=True)
class ProxmoxInventory:
    """One read of an account's Proxmox: its nodes and its guests."""

    nodes: tuple[str, ...]
    guests: tuple[GuestRecord, ...]


@dataclass(frozen=True)
class Resolution:
    """What inventory says about the target a request names."""

    status: str                      # resolved | ambiguous | not_found | unavailable
    arguments: dict[str, list[str]]  # grounded values to bind, as the routing contract binds them
    guest: GuestRecord | None = None
    node: str | None = None
    question: str | None = None      # the owner's choice, when one is needed
    grounding_line: str | None = None
    note: str | None = None          # what Janus found, said in the answer


class InventoryUnavailable(RuntimeError):
    """Inventory cannot be read; the message is safe to show the owner."""


def _words(text: str) -> list[str]:
    return TOKEN.findall((text or "").casefold())


def _name_words(guest: GuestRecord) -> set[str]:
    """The distinctive words of a guest's name and tags ("janus-lab" -> {"janus", "lab"})."""
    words = set()
    for part in (guest.name or "", *guest.tags):
        for word in _words(part):
            # A word that is only digits or a single letter names nothing on its own.
            if len(word) > 1 and not word.isdigit():
                words.add(word)
    for family in WORD_FAMILIES:
        if words & family:
            words |= family
    return words


def _named_in(name: str, text: str) -> bool:
    """Whether the exact name appears in the (casefolded) text as a whole name, not inside a longer one."""
    return bool(name) and re.search(
        rf"(?<![a-z0-9._-]){re.escape(name.casefold())}(?![a-z0-9_-])(?!\.[a-z0-9])", text) is not None


def _distinct(guests: Iterable[GuestRecord]) -> list[GuestRecord]:
    return sorted({guest.vmid: guest for guest in guests}.values(),
                  key=lambda guest: (guest.node.casefold(), guest.vmid))


def _described(text: str, guests: list[GuestRecord]) -> list[GuestRecord]:
    """The guests whose name or tag words best match the request's distinctive words."""
    request_words = {word for word in _words(text) if word not in GENERIC_WORDS and not word.isdigit()}
    if not request_words:
        return []
    scored = [(len(request_words & _name_words(guest)), guest) for guest in guests]
    best = max((score for score, _ in scored), default=0)
    return [guest for score, guest in scored if best and score == best]


def match_guests(request: str, guests: Iterable[GuestRecord], *, vmids: Iterable[str] = (),
                 excluded_vmids: Iterable[str] = ()) -> list[GuestRecord]:
    """The guests a request names: by exact name, by VMID, or by the distinctive words of a name.

    An exact name wins over descriptive words. A VMID and a description that agree narrow each other;
    a VMID and a name (or description) that point at different guests are all returned, so the owner
    chooses. Generic words ("container", "vm", "the node") name nothing, and a number the user excluded
    is never read as a target.
    """
    guests = list(guests)
    text = (request or "").casefold()
    if not text.strip() or not guests:
        return []
    excluded = set(excluded_vmids)
    numbers = {match for match in NUMBER.findall(request or "") if match not in excluded} | set(vmids)
    numbered = [guest for guest in guests if str(guest.vmid) in numbers]
    exact = [guest for guest in guests if guest.name and _named_in(guest.name, text)]
    if exact:
        return _distinct([*exact, *numbered])
    described = _described(text, guests)
    if numbered and described:
        agreeing = [guest for guest in numbered if guest in described]
        return _distinct(agreeing or [*numbered, *described])
    return _distinct(numbered or described)


def resolution_line(guest: GuestRecord) -> str:
    """The trusted line added to the grounding text: the exact values Proxmox lists for this guest."""
    kind = "VM" if guest.type == "qemu" else "LXC"
    return (f"[Janus resolved from live Proxmox inventory] the guest meant is {kind} {guest.vmid} "
            f"({guest.name or 'unnamed'}) on node {guest.node}.")


def _node_line(node: str) -> str:
    return f"[Janus resolved from live Proxmox inventory] the node meant is {node}."


def _listing(guests: list[GuestRecord]) -> str:
    shown = guests[:MAX_LISTED]
    lines = "".join(f"\n- {guest.label}" for guest in shown)
    rest = len(guests) - len(shown)
    return lines + (f"\n(and {rest} more)" if rest > 0 else "")


def _scope(tool: dict, missing: set[str]) -> str | None:
    """Which identity the missing values are: a guest's, a node's, or none this module resolves."""
    grounded = set(tool.get("grounded_arguments") or ())
    required = set(tool.get("required_arguments") or ())
    if not missing or not missing <= {"node", "vmid"} or not missing <= grounded:
        return None
    if {"node", "vmid"} <= grounded and {"node", "vmid"} <= required:
        return "guest"
    if missing == {"node"} and grounded == {"node"} and required == {"node"}:
        return "node"
    # A creation's node and VMID are where something new goes; a disk, a snapshot or a template is
    # never looked up here.
    return None


def resolvable(tool: dict, missing: Iterable[str]) -> bool:
    """Whether a step's missing values are an identity this module looks up (before reading inventory)."""
    return _scope(tool, set(missing or ())) is not None


def _describes_a_guest(value: str, guests: list[GuestRecord]) -> bool:
    """Whether a value is a guest's name or words of one (a description put into an identity argument)."""
    folded = value.casefold()
    if any(guest.name and guest.name.casefold() == folded for guest in guests):
        return True
    words = {word for word in _words(value) if word not in GENERIC_WORDS and not word.isdigit()}
    return bool(words) and any(words <= _name_words(guest) for guest in guests)


def resolve_proxmox_target(request: str, *, tool: dict, missing: Iterable[str],
                           inventory: Callable[[], ProxmoxInventory | Iterable[GuestRecord]],
                           bound: dict[str, list[str]] | None = None,
                           forbidden: dict[str, list[str]] | None = None) -> Resolution | None:
    """Resolve a missing guest (node and VMID) or node from live inventory, or None when not applicable.

    Applies only when every missing required value is part of the guest identity (``node`` and ``vmid``
    of a guest-scoped capability, or the ``node`` of a node-scoped one) and the capability grounds those
    arguments. Anything else the request left out stays a normal clarification.

    ``bound`` holds the values the routing contract kept because the request names them literally. A real
    VMID or node narrows the match and is never replaced; a value that is no VMID or node at all (a guest's
    name copied into the wrong argument) is only part of the user's description. ``forbidden`` holds what
    the user excluded: such a guest or node is never chosen or offered.
    """
    missing = set(missing)
    wanted = _scope(tool, missing)
    if wanted is None:
        return None

    try:
        read = inventory()
        if isinstance(read, ProxmoxInventory):
            all_guests, listed_nodes = list(read.guests), set(read.nodes)
        else:
            all_guests, listed_nodes = list(read), set()
    except InventoryUnavailable as error:
        return Resolution(status="unavailable", arguments={}, question=f"{error} Nothing was proposed.")
    except Exception:
        return Resolution(status="unavailable", arguments={}, question=(
            "Janus could not read your Proxmox inventory to find what you meant, so nothing was proposed. "
            "Name the guest's VMID and node, or try again when Proxmox answers."))

    forbidden = {name: {str(value).casefold() for value in values} for name, values in (forbidden or {}).items()}
    excluded_nodes, excluded_vmids = forbidden.get("node", set()), forbidden.get("vmid", set())
    nodes = sorted({node for node in (*listed_nodes, *(guest.node for guest in all_guests))
                    if node.casefold() not in excluded_nodes}, key=str.casefold)
    guests = [guest for guest in all_guests
              if guest.node.casefold() not in excluded_nodes and str(guest.vmid) not in excluded_vmids]
    text = (request or "").casefold()

    if wanted == "node":
        named = [node for node in nodes if _named_in(node, text)]
        if len(named) == 1:
            return Resolution(status="resolved", arguments={"node": named}, node=named[0],
                              grounding_line=_node_line(named[0]),
                              note=f"Janus found node {named[0]} in your Proxmox inventory.")
        if not named and len(nodes) == 1:
            return Resolution(status="resolved", arguments={"node": nodes}, node=nodes[0],
                              grounding_line=_node_line(nodes[0]),
                              note=f"Your Proxmox has one node, {nodes[0]}; Janus used it.")
        hosts = match_guests(request, guests, excluded_vmids=excluded_vmids) if not named else []
        if len({guest.node for guest in hosts}) == 1:
            node = hosts[0].node
            return Resolution(status="resolved", arguments={"node": [node]}, node=node,
                              grounding_line=_node_line(node),
                              note=f"Janus found node {node} in your Proxmox inventory: it runs "
                                   + ", ".join(guest.label for guest in hosts[:MAX_LISTED]) + ".")
        if not nodes:
            return Resolution(status="not_found", arguments={}, question="Your Proxmox lists no node.")
        return Resolution(status="ambiguous", arguments={}, question=(
            "Which node do you mean? Your Proxmox has: " + ", ".join(named or nodes) + "."))

    # Values the request binds literally narrow the candidates and are never replaced; a value that is
    # no real identity is the user's description of the guest, and anything else is not found.
    bound = {name: [str(value) for value in values] for name, values in (bound or {}).items() if values}
    known_nodes = {node.casefold() for node in nodes}
    bound_vmids, bound_nodes, keep = set(), set(), set()
    for value in bound.get("vmid", ()):
        if value.isdigit():
            bound_vmids.add(value)
        elif not _describes_a_guest(value, all_guests):
            guests = []
    for value in bound.get("node", ()):
        if value.casefold() in known_nodes:
            bound_nodes.add(value.casefold())
        elif not _describes_a_guest(value, all_guests):
            guests = []
    if bound_vmids:
        keep.add("vmid")
        if not any(str(guest.vmid) in bound_vmids for guest in guests):
            guests = []
    if bound_nodes:
        keep.add("node")
        guests = [guest for guest in guests if guest.node.casefold() in bound_nodes]

    # A guest named exactly but also excluded: say so instead of matching the rest of its words elsewhere.
    named_but_excluded = [guest for guest in all_guests if guest.name and _named_in(guest.name, text)
                          and (str(guest.vmid) in excluded_vmids or guest.node.casefold() in excluded_nodes)]
    named_and_allowed = [guest for guest in guests if (guest.name and _named_in(guest.name, text))
                         or str(guest.vmid) in bound_vmids
                         or str(guest.vmid) in set(NUMBER.findall(request or "")) - excluded_vmids]
    if named_but_excluded and not named_and_allowed:
        return Resolution(status="not_found", arguments={}, question=(
            "The guest you named is one you also excluded, so Janus proposes nothing: "
            + ", ".join(guest.label for guest in _distinct(named_but_excluded)) + ". Which guest do you mean?"))
    matches = match_guests(request, guests, vmids=bound_vmids, excluded_vmids=excluded_vmids)
    if len(matches) == 1:
        guest = matches[0]
        arguments = {name: [value] for name, value in (("node", guest.node), ("vmid", str(guest.vmid)))
                     if name not in keep}
        return Resolution(status="resolved", arguments=arguments, guest=guest, grounding_line=resolution_line(guest),
                          note=f"Janus found {guest.label} in your Proxmox inventory.")
    if matches:
        return Resolution(status="ambiguous", arguments={}, question=(
            f"{len(matches)} guests match what you asked for. Which one do you mean?" + _listing(matches)))
    listed = _distinct(guest for guest in all_guests if guest.node.casefold() not in excluded_nodes
                       and str(guest.vmid) not in excluded_vmids
                       and (not bound_nodes or guest.node.casefold() in bound_nodes))
    if not listed:
        return Resolution(status="not_found", arguments={}, question=(
            "No Proxmox guest matches what you asked for, and your Proxmox lists no other guest there."))
    return Resolution(status="not_found", arguments={}, question=(
        "No Proxmox guest matches what you asked for. These exist:" + _listing(listed)))


def resolves_target(tool_name: str) -> bool:
    """Whether a capability's target is looked up in Proxmox inventory (every Proxmox capability)."""
    from tools.proxmox.catalog import PROXMOX_TOOLS

    return tool_name in PROXMOX_TOOLS


def proxmox_inventory() -> ProxmoxInventory:
    """The requesting account's nodes and guests from one cluster read (its own link, else the default)."""
    from config import Settings
    from services.provider_links import owner_proxmox_service

    try:
        service = owner_proxmox_service(Settings())
    except ValueError as error:
        raise InventoryUnavailable("Proxmox VE has an invalid configuration on this Janus host.") from error
    if service is None:
        raise InventoryUnavailable("Proxmox VE is not connected for this account. Add a Proxmox link in "
                                   "Connections.")
    try:
        data = service.guest_inventory()["data"]
    finally:
        service.close()
    return ProxmoxInventory(
        nodes=tuple(data["nodes"]),
        guests=tuple(GuestRecord(vmid=row["vmid"], node=row["node"], type=row["type"], name=row["name"],
                                 status=row["status"], tags=tuple(row.get("tags") or ()))
                     for row in data["guests"]))
