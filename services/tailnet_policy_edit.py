"""Edit the tailnet policy file as text, one typed grant at a time (direction note D2).

The policy is HuJSON: JSON with comments and trailing commas. Re-serialising it would drop the
owner's comments, so an edit inserts or removes exactly one grant object in the text and leaves
every other byte as it was. A grant is typed (sources, destinations, ports), never free text, and
may name only tags, groups, autogroups and "*" (destinations): no users, addresses or host aliases,
which would put personal data into the approval and the model's arguments.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

SELECTOR = re.compile(r"(?:tag|group):[a-z0-9][a-z0-9-]{0,62}")
AUTOGROUPS = frozenset({"autogroup:member", "autogroup:self", "autogroup:admin", "autogroup:tagged",
                        "autogroup:owner", "autogroup:it-admin", "autogroup:network-admin"})
PORT = re.compile(r"(?:(?:tcp|udp|icmp):)?(?:\*|[0-9]{1,5}(?:-[0-9]{1,5})?)")
MAX_ITEMS = 20
MAX_POLICY_CHARS = 256 * 1024


class PolicyEditRefused(ValueError):
    """The edit cannot be made as asked; nothing was changed."""


@dataclass(frozen=True)
class GrantSpec:
    sources: tuple[str, ...]
    destinations: tuple[str, ...]
    ports: tuple[str, ...]

    def validated(self) -> "GrantSpec":
        for label, values in (("source", self.sources), ("destination", self.destinations),
                              ("port", self.ports)):
            if not isinstance(values, (tuple, list)) or not values or len(values) > MAX_ITEMS:
                raise PolicyEditRefused(f"Give between 1 and {MAX_ITEMS} {label}s.")
            if not all(isinstance(value, str) for value in values) or len(set(values)) != len(values):
                raise PolicyEditRefused(f"Each {label} must be a distinct name.")
        for value in self.sources + self.destinations:
            if "@" in value:
                raise PolicyEditRefused("A grant names tags, groups or autogroups, never a user.")
            if re.fullmatch(r"[0-9a-fA-F:.]+(?:/[0-9]{1,3})?", value) and any(ch.isdigit() for ch in value):
                raise PolicyEditRefused("A grant names tags, groups or autogroups, never an address.")
        for value in self.sources:
            if value == "*":
                raise PolicyEditRefused("A source must be a tag, group or autogroup, not every source.")
            if not (SELECTOR.fullmatch(value) or value in AUTOGROUPS):
                raise PolicyEditRefused(f"Unknown source selector {value[:40]!r}.")
        for value in self.destinations:
            if not (value == "*" or SELECTOR.fullmatch(value) or value in AUTOGROUPS):
                raise PolicyEditRefused(f"Unknown destination selector {value[:40]!r}.")
        for value in self.ports:
            if not PORT.fullmatch(value):
                raise PolicyEditRefused(f"Unknown port {value[:20]!r}.")
            for number in re.findall(r"[0-9]+", value.split(":")[-1]):
                if not 0 < int(number) <= 65535:
                    raise PolicyEditRefused(f"The port {number} is outside 1-65535.")
        return GrantSpec(tuple(self.sources), tuple(self.destinations), tuple(self.ports))

    def as_rule(self) -> dict:
        return {"src": list(self.sources), "dst": list(self.destinations), "ip": list(self.ports)}

    def matches(self, rule) -> bool:
        if not isinstance(rule, dict) or set(rule) != {"src", "dst", "ip"}:
            return False
        try:
            return (set(rule["src"]), set(rule["dst"]), set(rule["ip"])) == \
                (set(self.sources), set(self.destinations), set(self.ports))
        except TypeError:
            return False


def grant_text(spec: GrantSpec) -> str:
    """One canonical line: the approval shows it and a removal is confirmed by typing it."""
    return (", ".join(sorted(spec.sources)) + " -> " + ", ".join(sorted(spec.destinations)) + " : "
            + ", ".join(sorted(spec.ports)))


# --------------------------------------------------------------------------------------------
# A small HuJSON scanner: values with their exact text spans, comments and trailing commas.
# --------------------------------------------------------------------------------------------

@dataclass
class _Node:
    value: object
    start: int
    end: int
    items: list | None = None          # arrays: child nodes
    members: list | None = None        # objects: (key, key_start, child node)


class _Scanner:
    def __init__(self, text: str):
        self.text = text
        self.at = 0

    def fail(self) -> PolicyEditRefused:
        return PolicyEditRefused("The tailnet policy is not a policy object Janus can edit.")

    def skip(self) -> None:
        text = self.text
        while self.at < len(text):
            char = text[self.at]
            if char in " \t\r\n":
                self.at += 1
            elif text.startswith("//", self.at):
                end = text.find("\n", self.at)
                self.at = len(text) if end < 0 else end + 1
            elif text.startswith("/*", self.at):
                end = text.find("*/", self.at + 2)
                if end < 0:
                    raise self.fail()
                self.at = end + 2
            else:
                return

    def value(self) -> _Node:
        self.skip()
        if self.at >= len(self.text):
            raise self.fail()
        char = self.text[self.at]
        if char == "{":
            return self.obj()
        if char == "[":
            return self.arr()
        if char == '"':
            start = self.at
            return _Node(self.string(), start, self.at)
        match = re.compile(r"-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?|true|false|null").match(self.text, self.at)
        if not match:
            raise self.fail()
        self.at = match.end()
        return _Node(json.loads(match.group()), match.start(), match.end())

    def string(self) -> str:
        match = re.compile(r'"(?:[^"\\\n]|\\.)*"').match(self.text, self.at)
        if not match:
            raise self.fail()
        self.at = match.end()
        return json.loads(match.group())

    def arr(self) -> _Node:
        start = self.at
        self.at += 1
        items = []
        while True:
            self.skip()
            if self.at < len(self.text) and self.text[self.at] == "]":
                self.at += 1
                return _Node([item.value for item in items], start, self.at, items=items)
            items.append(self.value())
            self.skip()
            if self.at < len(self.text) and self.text[self.at] == ",":
                self.at += 1
            elif not (self.at < len(self.text) and self.text[self.at] == "]"):
                raise self.fail()

    def obj(self) -> _Node:
        start = self.at
        self.at += 1
        members = []
        while True:
            self.skip()
            if self.at < len(self.text) and self.text[self.at] == "}":
                self.at += 1
                keys = [key for key, _, _ in members]
                if len(set(keys)) != len(keys):
                    raise self.fail()
                return _Node({key: node.value for key, _, node in members}, start, self.at, members=members)
            if self.at >= len(self.text) or self.text[self.at] != '"':
                raise self.fail()
            key_start = self.at
            key = self.string()
            self.skip()
            if self.at >= len(self.text) or self.text[self.at] != ":":
                raise self.fail()
            self.at += 1
            members.append((key, key_start, self.value()))
            self.skip()
            if self.at < len(self.text) and self.text[self.at] == ",":
                self.at += 1
            elif not (self.at < len(self.text) and self.text[self.at] == "}"):
                raise self.fail()


def _document(text: str) -> _Node:
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_POLICY_CHARS:
        raise PolicyEditRefused("The tailnet policy is not a policy object Janus can edit.")
    scanner = _Scanner(text)
    root = scanner.value()
    scanner.skip()
    if scanner.at != len(text) or not isinstance(root.value, dict):
        raise scanner.fail()
    grants = root.value.get("grants")
    if grants is not None and not isinstance(grants, list):
        raise scanner.fail()
    return root


def parse_hujson(text: str) -> dict:
    value = _document(text).value
    assert isinstance(value, dict)
    return value


def _grants(root: _Node) -> _Node | None:
    for key, _, node in root.members or []:
        if key == "grants":
            return node
    return None


def _line_start(text: str, at: int) -> int:
    return text.rfind("\n", 0, at) + 1


def _indent(text: str, at: int) -> str:
    start = _line_start(text, at)
    match = re.match(r"[ \t]*", text[start:at])
    return match.group() if match else ""


def _rule_text(spec: GrantSpec) -> str:
    return json.dumps(spec.as_rule(), separators=(", ", ": "))


def add_grant(text: str, spec: GrantSpec) -> str:
    spec = spec.validated()
    root = _document(text)
    grants = _grants(root)
    if grants is not None and any(spec.matches(item.value) for item in grants.items or []):
        raise PolicyEditRefused(f"The grant {grant_text(spec)} is already in the policy.")
    if grants is None:
        # A new section just before the root's closing brace, with the root's member indent.
        close = root.end - 1
        member_indent = _indent(text, root.members[0][1]) if root.members else "\t"
        before = text[:close].rstrip()
        comma = "" if before.endswith((",", "{")) else ","
        edited = (before + comma + "\n" + member_indent + '"grants": [\n' + member_indent * 2 +
                  _rule_text(spec) + ",\n" + member_indent + "],\n" + text[close:])
    else:
        close = grants.end - 1
        indent = _indent(text, grants.items[-1].start) if grants.items else _indent(text, grants.start) + "\t"
        if grants.items:
            last = grants.items[-1]
            between = text[last.end:close]
            has_comma = between.lstrip().startswith(",")
            insert_at = last.end + (between.index(",") + 1 if has_comma else 0)
            prefix = "" if has_comma else ","
            edited = text[:insert_at] + prefix + "\n" + indent + _rule_text(spec) + ("," if has_comma else "") + \
                text[insert_at:]
        else:
            edited = text[:close] + "\n" + indent + _rule_text(spec) + ",\n" + _indent(text, grants.start) + \
                text[close:]
    _confirm_one_grant_changed(text, edited, spec, added=True)
    return edited


def remove_grant(text: str, spec: GrantSpec) -> str:
    spec = spec.validated()
    root = _document(text)
    grants = _grants(root)
    found = [item for item in ((grants.items or []) if grants is not None else []) if spec.matches(item.value)]
    if not found:
        raise PolicyEditRefused(f"There is no grant {grant_text(spec)} in the policy.")
    if len(found) > 1:
        raise PolicyEditRefused(f"The policy has more than one grant {grant_text(spec)}; edit it by hand.")
    item = found[0]
    # The object's own line(s), from the start of its line through its comma and line end.
    start = _line_start(text, item.start)
    if text[start:item.start].strip():
        start = item.start
    end = item.end
    rest = text[end:]
    comma = re.match(r"[ \t]*,", rest)
    if comma:
        end += comma.end()
    newline = re.match(r"[ \t]*\r?\n", text[end:])
    if newline and start == _line_start(text, item.start):
        end += newline.end()
    edited = text[:start] + text[end:]
    _confirm_one_grant_changed(text, edited, spec, added=False)
    return edited


def _confirm_one_grant_changed(before_text: str, after_text: str, spec: GrantSpec, *, added: bool) -> None:
    """The parsed policy differs by exactly that grant; everything else is unchanged."""
    before, after = parse_hujson(before_text), parse_hujson(after_text)
    old, new = list(before.get("grants") or []), list(after.get("grants") or [])
    expected = old + [spec.as_rule()] if added else [rule for rule in old if not spec.matches(rule)]
    others_same = {key: value for key, value in before.items() if key != "grants"} == \
        {key: value for key, value in after.items() if key != "grants"}
    if new != expected or not others_same:
        raise PolicyEditRefused("Janus could not make exactly that one change to the policy text.")
