"""Human context: a Proxmox guest named the way people name it is found in live inventory.

Owner decision, 26 November 2026 (direction note): "Look it up in live inventory: one clear match is
proposed with its real ID and name for your approval; several matches, you pick from a short list; none,
it says so". The identity comes from Proxmox's own listing, never from a model; the request text the
gateway grounds against gains a trusted line naming the exact guest, so every later check (binding,
SemanticGuard, ToolGateway, the policy freeze, approval and the typed name for a destructive change) runs
unchanged on real values.
"""
from __future__ import annotations

import pytest

from subagents.core.orchestration.target_resolution import (
    GuestRecord,
    match_guests,
    resolution_line,
    resolve_proxmox_target,
)

GUESTS = (
    GuestRecord(vmid=990, node="hummingbird", type="lxc", name="janus-lab", status="stopped", tags=("janus",)),
    GuestRecord(vmid=991, node="hummingbird", type="lxc", name="janus-trainer", status="running",
                tags=("janus", "trainer")),
    GuestRecord(vmid=992, node="hummingbird", type="lxc", name="janus-control", status="running", tags=()),
    GuestRecord(vmid=100, node="hummingbird", type="lxc", name="matrix-11200648", status="stopped", tags=()),
    GuestRecord(vmid=120, node="blackhawk", type="qemu", name="db-postgres", status="running",
                tags=("db-postgresql",)),
)


@pytest.mark.parametrize("request_text,expected", [
    # The exact name, in any case, anywhere in the sentence.
    ("restart janus-lab please", [990]),
    ("Restart JANUS-LAB", [990]),
    # The name's distinctive words, as people say them.
    ("start the lab container", [990]),
    ("is the trainer up?", [991]),
    ("shut down the control plane box", [992]),
    ("check the postgres database vm", [120]),
    # A VMID alone is enough; the node comes from inventory.
    ("delete guest 100", [100]),
    ("reboot 991", [991]),
])
def test_one_clear_match_is_found_from_how_people_say_it(request_text, expected):
    assert [guest.vmid for guest in match_guests(request_text, GUESTS)] == expected


def test_words_shared_by_several_guests_give_every_candidate():
    # "janus" names three guests: the owner picks; nothing is chosen for them.
    assert sorted(guest.vmid for guest in match_guests("restart the janus container", GUESTS)) == [990, 991, 992]


def test_nothing_is_matched_from_generic_words_or_from_nothing():
    for text in ("restart the container", "restart it", "start the vm on the node", "show me everything",
                 "reboot guest 4242", ""):
        assert match_guests(text, GUESTS) == [], text


def test_a_number_inside_another_word_or_value_is_not_a_vmid():
    # "1024 MiB", "v100", "port 9910" and "10.0.0.100" do not name guest 100 or 991.
    for text in ("give it 1024 MiB", "the v100 card", "port 9910 is open", "ping 10.0.0.100"):
        assert match_guests(text, GUESTS) == [], text


def test_the_resolution_line_names_the_exact_guest_for_the_gateway():
    guest = GUESTS[0]
    line = resolution_line(guest)
    # Literal values the gateway can ground: the node and the VMID as Proxmox lists them.
    assert "hummingbird" in line and "990" in line and "janus-lab" in line
    assert line.startswith("[Janus resolved from live Proxmox inventory]")


class Inventory:
    def __init__(self, guests=GUESTS, fail=None):
        self.guests, self.fail, self.calls = guests, fail, 0

    def __call__(self):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return self.guests


GUEST_TOOL = {"grounded_arguments": ["node", "vmid"], "required_arguments": ["node", "vmid"]}


def test_a_missing_guest_identity_with_one_match_becomes_the_exact_target():
    inventory = Inventory()
    outcome = resolve_proxmox_target(
        "please restart the lab container", tool=GUEST_TOOL, missing=["node", "vmid"], inventory=inventory)
    assert outcome.status == "resolved"
    assert outcome.guest == GUESTS[0]
    assert outcome.arguments == {"node": ["hummingbird"], "vmid": ["990"]}
    assert inventory.calls == 1


def test_several_matches_ask_with_a_short_list_and_choose_nothing():
    outcome = resolve_proxmox_target(
        "restart the janus container", tool=GUEST_TOOL, missing=["node", "vmid"], inventory=Inventory())
    assert outcome.status == "ambiguous"
    assert outcome.arguments == {}
    text = outcome.question
    for name, vmid in (("janus-lab", 990), ("janus-trainer", 991), ("janus-control", 992)):
        assert name in text and str(vmid) in text
    assert "Which one" in text


def test_no_match_says_so_and_lists_what_exists():
    outcome = resolve_proxmox_target(
        "restart the minecraft server", tool=GUEST_TOOL, missing=["node", "vmid"], inventory=Inventory())
    assert outcome.status == "not_found"
    assert outcome.arguments == {}
    assert "No Proxmox guest" in outcome.question
    assert "janus-lab" in outcome.question


def test_a_long_candidate_list_is_cut_short():
    many = tuple(GuestRecord(vmid=200 + index, node="hummingbird", type="lxc", name=f"web-{index}",
                             status="running", tags=()) for index in range(30))
    outcome = resolve_proxmox_target(
        "restart the web container", tool=GUEST_TOOL, missing=["node", "vmid"], inventory=Inventory(many))
    assert outcome.status == "ambiguous"
    assert outcome.question.count("\n- ") <= 8
    assert "22 more" in outcome.question


def test_only_the_guest_identity_is_ever_resolved():
    # A snapshot name, a size, a template or a storage is never looked up: those stay the user's words.
    snapshot_tool = {"grounded_arguments": ["node", "vmid"], "required_arguments": ["node", "vmid", "snapshot"]}
    outcome = resolve_proxmox_target("roll back the lab container", tool=snapshot_tool,
                                     missing=["snapshot"], inventory=Inventory())
    assert outcome is None
    template_tool = {"grounded_arguments": ["node", "template", "vmid"], "required_arguments": ["node", "template"]}
    assert resolve_proxmox_target("create a lab container", tool=template_tool,
                                  missing=["node", "template"], inventory=Inventory()) is None


def test_a_node_only_tool_resolves_the_one_node_named():
    node_tool = {"grounded_arguments": ["node"], "required_arguments": ["node"]}
    outcome = resolve_proxmox_target("how loaded is the hummingbird node", tool=node_tool, missing=["node"],
                                     inventory=Inventory())
    assert outcome.status == "resolved" and outcome.arguments == {"node": ["hummingbird"]}
    # One node in the whole cluster: \"the node\" can only mean it.
    single = tuple(guest for guest in GUESTS if guest.node == "hummingbird")
    outcome = resolve_proxmox_target("how loaded is the node", tool=node_tool, missing=["node"],
                                     inventory=Inventory(single))
    assert outcome.status == "resolved" and outcome.arguments == {"node": ["hummingbird"]}
    # Two nodes, none named: ask.
    outcome = resolve_proxmox_target("how loaded is the node", tool=node_tool, missing=["node"],
                                     inventory=Inventory())
    assert outcome.status == "ambiguous" and "blackhawk" in outcome.question


def test_values_the_request_already_bound_narrow_the_match_and_are_never_replaced():
    # "restart 991" binds the VMID literally; inventory only adds the node it lives on.
    outcome = resolve_proxmox_target("restart 991", tool=GUEST_TOOL, missing=["node"], inventory=Inventory(),
                                     bound={"vmid": ["991"]})
    assert outcome.status == "resolved" and outcome.arguments == {"node": ["hummingbird"]}
    # A bound VMID that contradicts the guest named in the same request is never silently preferred: ask.
    outcome = resolve_proxmox_target("restart janus-lab, VMID 100", tool=GUEST_TOOL, missing=["node"],
                                     inventory=Inventory(), bound={"vmid": ["100"]})
    assert outcome.status == "ambiguous" and outcome.arguments == {}
    assert "janus-lab" in outcome.question and "matrix-11200648" in outcome.question
    # A name and a VMID that agree resolve.
    outcome = resolve_proxmox_target("restart 991 (janus-trainer)", tool=GUEST_TOOL, missing=["node"],
                                     inventory=Inventory(), bound={"vmid": ["991"]})
    assert outcome.status == "resolved" and outcome.guest.vmid == 991
    # A bound VMID no guest has is not found, whatever else the request says.
    outcome = resolve_proxmox_target("restart the lab container 4242", tool=GUEST_TOOL, missing=["node"],
                                     inventory=Inventory(), bound={"vmid": ["4242"]})
    assert outcome.status == "not_found" and outcome.arguments == {}
    # A bound node keeps the search on that node.
    outcome = resolve_proxmox_target("restart the postgres vm on hummingbird", tool=GUEST_TOOL, missing=["vmid"],
                                     inventory=Inventory(), bound={"node": ["hummingbird"]})
    assert outcome.status == "not_found"


def test_a_value_bound_to_the_wrong_argument_is_read_as_the_users_description():
    # A model that copies "lab" into vmid or node did not name a VMID or a node: the words describe the guest.
    outcome = resolve_proxmox_target("restart the lab container", tool=GUEST_TOOL, missing=["node"],
                                     inventory=Inventory(), bound={"vmid": ["lab"]})
    assert outcome.status == "resolved"
    assert outcome.arguments == {"node": ["hummingbird"], "vmid": ["990"]}
    outcome = resolve_proxmox_target("restart janus-lab", tool=GUEST_TOOL, missing=["vmid"],
                                     inventory=Inventory(), bound={"node": ["janus-lab"]})
    assert outcome.status == "resolved"
    assert outcome.arguments == {"node": ["hummingbird"], "vmid": ["990"]}


def test_a_value_the_user_excluded_is_never_chosen():
    outcome = resolve_proxmox_target("restart the janus containers but not 990", tool=GUEST_TOOL,
                                     missing=["node", "vmid"], inventory=Inventory(),
                                     forbidden={"vmid": ["990"]})
    assert outcome.status == "ambiguous"
    assert "janus-lab" not in outcome.question
    outcome = resolve_proxmox_target("restart janus-lab, never 990", tool=GUEST_TOOL, missing=["node", "vmid"],
                                     inventory=Inventory(), forbidden={"vmid": ["990"]})
    # The guest named is the one the owner excluded: say exactly that, propose nothing.
    assert outcome.status == "not_found" and outcome.arguments == {}
    assert "janus-lab" in outcome.question and "excluded" in outcome.question


def test_node_lookup_is_only_for_node_scoped_capabilities():
    # A creation tool's node is where something new goes: never looked up.
    create = {"grounded_arguments": ["engine", "node", "version", "vmid"],
              "required_arguments": ["engine", "hostname", "node", "version"]}
    assert resolve_proxmox_target("create a postgres database", tool=create, missing=["node"],
                                  inventory=Inventory()) is None
    smart = {"grounded_arguments": ["disk", "node"], "required_arguments": ["disk", "node"]}
    assert resolve_proxmox_target("smart for sda", tool=smart, missing=["node"], inventory=Inventory()) is None


def test_an_unreadable_inventory_never_guesses():
    outcome = resolve_proxmox_target("restart the lab container", tool=GUEST_TOOL, missing=["node", "vmid"],
                                     inventory=Inventory(fail=RuntimeError("Proxmox returned an invalid answer.")))
    assert outcome.status == "unavailable"
    assert outcome.arguments == {}
    assert "could not read" in outcome.question
