"""D2: the tailnet policy file is edited as text, one typed grant at a time (direction note D2).

The live policy is HuJSON: comments and trailing commas that an API write of plain JSON would drop.
An edit inserts or removes exactly one grant object in the text and leaves every other byte as it
was; the parsed result must differ from the original by exactly that grant. Only tags, groups,
autogroups and "*" may appear in a typed grant: never users, addresses or host aliases.
"""
import pytest

from services.tailnet_policy_edit import (
    GrantSpec,
    PolicyEditRefused,
    add_grant,
    grant_text,
    parse_hujson,
    remove_grant,
)

LIVE_LIKE = """// Example/default ACLs for unrestricted connections.
{
\t// Declare static groups of users. Use autogroups for all users or users with a specific role.
\t// "groups": {
\t//   "group:example": ["alice@example.com", "bob@example.com"],
\t// },

\t// Define the tags which can be applied to devices and by which users.
\t"tagOwners": {
\t\t"tag:janus-lab": ["autogroup:admin"],
\t},

\t// Define grants that govern access for users, groups, autogroups, tags,
\t// Tailscale IP addresses, and subnet ranges.
\t"grants": [
\t\t// Allow all connections.
\t\t// Comment this section out if you want to define specific restrictions.
\t\t{"src": ["autogroup:member"], "dst": ["*"], "ip": ["*"]},
\t],

\t// Define users and devices that can use Tailscale SSH.
\t"ssh": [
\t\t// Allow all users to SSH into their own devices in check mode.
\t\t{
\t\t\t"action": "check",
\t\t\t"src":    ["autogroup:member"],
\t\t\t"dst":    ["autogroup:self"],
\t\t\t"users":  ["autogroup:nonroot", "root"],
\t\t},
\t],
}
"""
LAB = GrantSpec(sources=("tag:janus-lab",), destinations=("tag:janus-lab",), ports=("22",))


def test_the_live_hujson_shape_parses_with_comments_and_trailing_commas():
    parsed = parse_hujson(LIVE_LIKE)
    assert parsed["grants"] == [{"src": ["autogroup:member"], "dst": ["*"], "ip": ["*"]}]
    assert parsed["ssh"][0]["users"] == ["autogroup:nonroot", "root"]
    assert parsed["tagOwners"] == {"tag:janus-lab": ["autogroup:admin"]}
    # A "//" inside a string is text, not a comment.
    assert parse_hujson('{"a": "http://x" // note\n}') == {"a": "http://x"}
    assert parse_hujson('{"a": [1, /* two */ 2,],}') == {"a": [1, 2]}


def test_adding_a_grant_keeps_every_comment_and_changes_exactly_one_grant():
    edited = add_grant(LIVE_LIKE, LAB)
    for line in LIVE_LIKE.splitlines():
        if line.strip().startswith("//"):
            assert line in edited, line
    before, after = parse_hujson(LIVE_LIKE), parse_hujson(edited)
    assert after["grants"] == before["grants"] + [{"src": ["tag:janus-lab"], "dst": ["tag:janus-lab"],
                                                     "ip": ["22"]}]
    assert {key: value for key, value in after.items() if key != "grants"} == \
        {key: value for key, value in before.items() if key != "grants"}
    # Everything before the grants' closing bracket that was there is still there, byte for byte.
    head = LIVE_LIKE[:LIVE_LIKE.index('{"src": ["autogroup:member"]')]
    assert edited.startswith(head)


def test_removing_the_grant_again_restores_the_original_text_exactly():
    assert remove_grant(add_grant(LIVE_LIKE, LAB), LAB) == LIVE_LIKE


def test_a_grant_is_matched_by_its_sets_not_its_order_or_spacing():
    text = LIVE_LIKE.replace('{"src": ["autogroup:member"], "dst": ["*"], "ip": ["*"]},',
                             '{"src": ["autogroup:member"], "dst": ["*"], "ip": ["*"]},\n'
                             '\t\t{ "ip": ["443", "22"],  "dst": ["tag:janus-lab"], "src": ["tag:janus-lab"] },')
    spec = GrantSpec(sources=("tag:janus-lab",), destinations=("tag:janus-lab",), ports=("22", "443"))
    removed = remove_grant(text, spec)
    assert parse_hujson(removed)["grants"] == [{"src": ["autogroup:member"], "dst": ["*"], "ip": ["*"]}]


def test_a_duplicate_add_and_a_missing_or_ambiguous_remove_are_refused():
    with pytest.raises(PolicyEditRefused, match="already"):
        add_grant(add_grant(LIVE_LIKE, LAB), LAB)
    with pytest.raises(PolicyEditRefused, match="no grant"):
        remove_grant(LIVE_LIKE, LAB)
    twice = LIVE_LIKE.replace('{"src": ["autogroup:member"], "dst": ["*"], "ip": ["*"]},',
                              '{"src": ["autogroup:member"], "dst": ["*"], "ip": ["*"]},\n'
                              '\t\t{"src": ["autogroup:member"], "dst": ["*"], "ip": ["*"]},')
    member_all = GrantSpec(sources=("autogroup:member",), destinations=("*",), ports=("*",))
    with pytest.raises(PolicyEditRefused, match="more than one"):
        remove_grant(twice, member_all)


def test_a_policy_without_grants_gets_a_grants_section():
    text = '{\n\t"tagOwners": {"tag:janus-lab": ["autogroup:admin"]},\n}\n'
    edited = add_grant(text, LAB)
    assert parse_hujson(edited)["grants"] == [{"src": ["tag:janus-lab"], "dst": ["tag:janus-lab"], "ip": ["22"]}]
    assert parse_hujson(edited)["tagOwners"] == {"tag:janus-lab": ["autogroup:admin"]}


@pytest.mark.parametrize("sources,destinations,ports,words", [
    (("alice@example.com",), ("*",), ("*",), "user"),
    (("tag:janus-lab",), ("100.64.0.1",), ("22",), "address"),
    (("tag:janus-lab",), ("nas",), ("22",), "selector"),
    (("*",), ("tag:janus-lab",), ("22",), "source"),
    (("tag:janus-lab",), ("tag:janus-lab",), ("70000",), "port"),
    (("tag:janus-lab",), ("tag:janus-lab",), ("22;rm",), "port"),
    ((), ("tag:janus-lab",), ("22",), "source"),
    (("tag:Bad Name",), ("tag:janus-lab",), ("22",), "selector"),
])
def test_only_tags_groups_autogroups_and_star_destinations_with_real_ports_are_accepted(
        sources, destinations, ports, words):
    with pytest.raises(PolicyEditRefused, match=words):
        GrantSpec(sources=sources, destinations=destinations, ports=ports).validated()


def test_the_grant_reads_the_same_way_in_the_approval_and_the_typed_confirmation():
    assert grant_text(LAB) == "tag:janus-lab -> tag:janus-lab : 22"
    spec = GrantSpec(sources=("group:ops", "autogroup:member"), destinations=("tag:db",), ports=("tcp:5432", "22"))
    # Sets are sorted so the same grant always reads the same way.
    assert grant_text(spec) == "autogroup:member, group:ops -> tag:db : 22, tcp:5432"


def test_text_that_is_not_a_policy_object_is_refused():
    for bad in ("", "[]", '{"grants": "all"}', '{"grants": [', '{"a": 1} trailing'):
        with pytest.raises(PolicyEditRefused):
            add_grant(bad, LAB)
