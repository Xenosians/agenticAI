import json
import re

import pytest

from services.directory import (
    LdapDirectoryService,
    MockDirectoryService,
)
from services.directory.base import (
    MAX_ACCOUNT_SEARCH_RESULTS,
    MAX_GROUP_MEMBERS,
    DirectoryService,
)


BASE_DN = "DC=itsm,DC=local"
DOMAIN_SID = "S-1-5-21-1-2-3"
IN_CHAIN = "1.2.840.113556.1.4.1941"
PAGED = "1.2.840.113556.1.4.319"


def dn(name, container="CN=Users"):
    return f"CN={name},{container},{BASE_DN}"


class FakeAttribute:

    def __init__(self, values):
        self.values = list(values)
        self.raw_values = [
            value if isinstance(value, bytes) else str(value).encode()
            for value in self.values
        ]
        self.value = (
            self.values[0]
            if len(self.values) == 1
            else (self.values or None)
        )

    def __bool__(self):
        return bool(self.values)


class FakeEntry:

    def __init__(self, entry_dn, attributes):
        self.entry_dn = entry_dn
        self._attributes = attributes

    def __getitem__(self, name):
        if name not in self._attributes:
            raise KeyError(name)
        return FakeAttribute(self._attributes[name])

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def user(name, *, display=None, mail=None, uac=512, lockout=0,
         member_of=(), rid=1100, primary=513):
    attributes = {
        "sAMAccountName": [name],
        "cn": [name],
        "objectClass": ["top", "person", "user"],
        "userAccountControl": [uac],
        "memberOf": list(member_of),
        "objectSid": [f"{DOMAIN_SID}-{rid}"],
        "primaryGroupID": [primary],
        # Credential attributes must never be requested or returned.
        "unicodePwd": ["secret-hash"],
    }
    if display:
        attributes["displayName"] = [display]
    if mail:
        attributes["mail"] = [mail]
    if lockout is not None:
        attributes["lockoutTime"] = [lockout]
    return FakeEntry(dn(name), attributes)


def group(name, rid, *, container="CN=Users", member_of=()):
    return FakeEntry(dn(name, container), {
        "cn": [name],
        "sAMAccountName": [name],
        "objectClass": ["top", "group"],
        "objectSid": [f"{DOMAIN_SID}-{rid}"],
        "memberOf": list(member_of),
    })


class FakeDirectory:
    """A tiny in-memory AD that evaluates the filters the service sends."""

    def __init__(self, entries, members):
        self.entries_by_dn = {entry.entry_dn.casefold(): entry for entry in entries}
        self.members = {key.casefold(): [m.casefold() for m in value] for key, value in members.items()}

    def entry(self, entry_dn):
        return self.entries_by_dn.get(entry_dn.casefold())

    def chain_groups(self, member_dn):
        found, frontier = set(), [member_dn.casefold()]
        while frontier:
            current = frontier.pop()
            for group_dn, values in self.members.items():
                if current in values and group_dn not in found:
                    found.add(group_dn)
                    frontier.append(group_dn)
        return found

    def chain_members(self, group_dn):
        found, frontier = set(), [group_dn.casefold()]
        while frontier:
            current = frontier.pop()
            for member in self.members.get(current, []):
                if member not in found:
                    found.add(member)
                    frontier.append(member)
        return found


def unescape(value):
    return re.sub(r"\\([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), value)


class FakeConnection:

    def __init__(self, directory, *, result_code=0):
        self.directory = directory
        self.result_code = result_code
        self.searches = []
        self.entries = []
        self.result = {}
        self.unbound = False

    def _select(self, search_base, search_filter, search_scope):
        d = self.directory
        everything = list(d.entries_by_dn.values())
        if search_scope == "BASE":
            entry = d.entry(search_base)
            return [entry] if entry is not None and "(objectClass=group)" == search_filter else []
        if search_filter.startswith("(&(objectClass=user)(sAMAccountName="):
            name = unescape(search_filter.split("sAMAccountName=", 1)[1][:-2])
            return [e for e in everything if "user" in e._attributes["objectClass"]
                    and e._attributes["sAMAccountName"][0].casefold() == name.casefold()]
        match = re.fullmatch(r"\(&\(objectClass=group\)\(member:" + IN_CHAIN + r":=(.+)\)\)", search_filter)
        if match:
            groups = d.chain_groups(unescape(match.group(1)))
            return [d.entry(g) for g in sorted(groups)]
        match = re.fullmatch(r"\(memberOf:" + IN_CHAIN + r":=(.+)\)", search_filter)
        if match:
            return [d.entry(m) for m in sorted(d.chain_members(unescape(match.group(1)))) if d.entry(m)]
        match = re.fullmatch(r"\(&\(objectClass=group\)\(objectSid=(.+)\)\)", search_filter)
        if match:
            sid = unescape(match.group(1))
            return [e for e in everything if "group" in e._attributes["objectClass"]
                    and e._attributes["objectSid"][0] == sid]
        match = re.fullmatch(r"\(primaryGroupID=(\d+)\)", search_filter)
        if match:
            return [e for e in everything if "user" in e._attributes["objectClass"]
                    and str(e._attributes["primaryGroupID"][0]) == match.group(1)]
        match = re.fullmatch(r"\(&\(objectClass=group\)(?:\(\|)?\(cn=([^)]*)\)(?:\(sAMAccountName=[^)]*\)\))?\)", search_filter)
        if match:
            name = unescape(match.group(1)).casefold()
            return [e for e in everything if "group" in e._attributes["objectClass"]
                    and name in {e._attributes["cn"][0].casefold(), e._attributes["sAMAccountName"][0].casefold()}]
        match = re.fullmatch(r"\(&\(objectClass=user\)\(!\(objectClass=computer\)\)\(\|\(sAMAccountName=\*(.+)\*\)\(displayName=\*.+\*\)\(mail=\*.+\*\)\)\)", search_filter)
        if match:
            needle = unescape(match.group(1)).casefold()
            return [e for e in everything if "user" in e._attributes["objectClass"] and any(
                needle in str(e._attributes.get(key, [""])[0]).casefold()
                for key in ("sAMAccountName", "displayName", "mail"))]
        raise AssertionError(f"Unexpected filter: {search_filter}")

    def search(self, *, search_base, search_filter, attributes, search_scope="SUBTREE",
               time_limit=0, paged_size=None, paged_cookie=None, size_limit=0):
        self.searches.append({"base": search_base, "filter": search_filter, "attributes": list(attributes),
                              "scope": search_scope, "paged_size": paged_size, "time_limit": time_limit})
        if self.result_code:
            self.entries = []
            self.result = {"result": self.result_code, "description": "busy"}
            return False
        if search_scope != "BASE" and not search_base.casefold().endswith(BASE_DN.casefold()):
            self.entries = []
            self.result = {"result": 32, "description": "noSuchObject"}
            return False
        selected = self._select(search_base, search_filter, search_scope)
        start = int(paged_cookie or 0)
        page = selected[start:start + paged_size] if paged_size else selected
        more = paged_size and start + paged_size < len(selected)
        self.entries = page
        self.result = {"result": 0, "description": "success",
                       "controls": {PAGED: {"value": {"cookie": str(start + paged_size).encode() if more else b""}}}}
        return bool(page)

    def unbind(self):
        self.unbound = True


class FakeLdapDirectoryService(LdapDirectoryService):

    def __init__(self, directory, *, access_groups=None, result_code=0):
        self.base_dn = BASE_DN
        self.access_groups = self._normalize_access_groups(
            access_groups if access_groups is not None
            else {"vpn": "VPN-Users", "admin": "ITSM-Admins", "wiki": dn("Wiki Readers")})
        self.connections = []
        self.directory = directory
        self.result_code = result_code

    def _connect(self):
        connection = FakeConnection(self.directory, result_code=self.result_code)
        self.connections.append(connection)
        return connection

    def _connect_write(self):
        raise AssertionError("Directory reads must never open a write connection.")


def lab(extra_users=()):
    """jdoe is a direct VPN member; mnested reaches VPN through a nested group."""
    users = [
        user("jdoe", member_of=[dn("VPN-Users")], rid=1101),
        user("asmith", lockout=133000000000000000, member_of=[dn("VPN-Users"), dn("ITSM-Admins")], rid=1102),
        user("disabled", uac=514, rid=1103),
        user("mnested", display="Matrix Nested", mail="mnested@itsm.local", member_of=[dn("Contractors")], rid=1104),
        *extra_users,
    ]
    groups = [
        group("VPN-Users", 1201),
        group("ITSM-Admins", 1202),
        group("Contractors", 1203, member_of=[dn("VPN-Users")]),
        group("Domain Users", 513, member_of=[dn("Users", "CN=Builtin")]),
        group("Users", 545, container="CN=Builtin"),
        group("Wiki Readers", 1204),
    ]
    members = {
        dn("VPN-Users"): [dn("jdoe"), dn("asmith"), dn("Contractors")],
        dn("ITSM-Admins"): [dn("asmith")],
        dn("Contractors"): [dn("mnested")],
        dn("Users", "CN=Builtin"): [dn("Domain Users")],
    }
    return FakeDirectory(users + groups, members)


def service(**kwargs):
    return FakeLdapDirectoryService(lab(kwargs.pop("extra_users", ())), **kwargs)


def assert_no_credentials(svc, result):
    text = json.dumps(result)
    assert "secret-hash" not in text and "unicodePwd" not in text
    for connection in svc.connections:
        assert connection.unbound
        for search in connection.searches:
            assert not {"unicodePwd", "userPassword", "ntPwdHistory"} & set(search["attributes"])
            # The reused exact-name lookup returns one entry; every other read is bounded.
            if not search["filter"].startswith("(&(objectClass=user)(sAMAccountName="):
                assert search["time_limit"] and search["paged_size"]


# ================================================================
# account_search
# ================================================================

def test_account_search_matches_username_display_name_and_email():
    svc = service()
    result = svc.account_search("Matrix")
    assert result["ok"] is True and result["count"] == 1 and result["truncated"] is False
    assert result["accounts"] == [{"username": "mnested", "display_name": "Matrix Nested",
                                   "email": "mnested@itsm.local", "enabled": True, "locked": False}]
    assert svc.account_search("itsm.local")["accounts"][0]["username"] == "mnested"
    assert_no_credentials(svc, result)


def test_account_search_reports_disabled_and_locked_state():
    svc = service()
    asmith = svc.account_search("asmith")["accounts"][0]
    disabled = svc.account_search("disabled")["accounts"][0]
    assert (asmith["enabled"], asmith["locked"]) == (True, True)
    assert (disabled["enabled"], disabled["locked"]) == (False, False)


def test_account_search_escapes_filter_characters():
    svc = service()
    result = svc.account_search("*)(sAMAccountName=*")
    assert result["ok"] is True and result["count"] == 0
    sent = svc.connections[0].searches[0]["filter"]
    assert "*)(sAMAccountName=*" not in sent and "\\2a\\29\\28sAMAccountName=\\2a" in sent


def test_account_search_is_capped():
    extra = [user(f"bulk{i:02d}", rid=2000 + i) for i in range(MAX_ACCOUNT_SEARCH_RESULTS + 3)]
    result = service(extra_users=extra).account_search("bulk")
    assert result["count"] == MAX_ACCOUNT_SEARCH_RESULTS and result["truncated"] is True


@pytest.mark.parametrize("query", ["", " ", "j", "x" * 65, None])
def test_account_search_rejects_unbounded_fragments_without_querying(query):
    svc = service()
    result = svc.account_search(query)
    assert result["ok"] is False and result["error_code"] == "invalid_argument"
    assert svc.connections == []


def test_provider_failure_is_reported_not_raised():
    svc = service(result_code=51)
    result = svc.account_search("jdoe")
    assert result["ok"] is False and result["error_code"] == "provider_error"
    assert "busy" in result["error"]
    assert svc.connections[0].unbound


# ================================================================
# user_groups
# ================================================================

def test_user_groups_include_direct_primary_and_nested_membership():
    svc = service()
    result = svc.user_groups("mnested")
    assert result["ok"] is True and result["user_id"] == "mnested"
    assert result["groups"] == [
        {"name": "Contractors", "membership": "direct", "resources": []},
        {"name": "Domain Users", "membership": "primary", "resources": []},
        {"name": "Users", "membership": "nested", "resources": []},
        {"name": "VPN-Users", "membership": "nested", "resources": ["vpn"]},
    ]
    chain = [s for c in svc.connections for s in c.searches if IN_CHAIN in s["filter"]]
    assert chain, "nested membership must use LDAP_MATCHING_RULE_IN_CHAIN"
    assert_no_credentials(svc, result)


def test_user_groups_mark_resources_mapped_by_name_or_dn():
    svc = service(access_groups={"vpn": "VPN-Users", "remote": dn("VPN-Users"), "admin": "ITSM-Admins"})
    groups = {item["name"]: item["resources"] for item in svc.user_groups("asmith")["groups"]}
    assert groups["VPN-Users"] == ["remote", "vpn"]
    assert groups["ITSM-Admins"] == ["admin"]


def test_user_groups_unknown_user_is_not_found():
    result = service().user_groups("nobody")
    assert result["ok"] is False and result["error_code"] == "not_found"


# ================================================================
# group_members
# ================================================================

def test_group_members_by_logical_resource_lists_direct_and_nested_members():
    svc = service()
    result = svc.group_members("vpn")
    assert result["ok"] is True and result["group"] == "VPN-Users" and result["resources"] == ["vpn"]
    assert [(m["name"], m["type"], m["membership"]) for m in result["members"]] == [
        ("asmith", "user", "direct"), ("Contractors", "group", "direct"),
        ("jdoe", "user", "direct"), ("mnested", "user", "nested"),
    ]
    assert_no_credentials(svc, result)


@pytest.mark.parametrize("name", ["VPN-Users", "vpn-users", "vpn access", "VPN group"])
def test_group_members_accepts_exact_group_names_and_resource_aliases(name):
    assert service().group_members(name)["group"] == "VPN-Users"


def test_group_members_include_primary_group_accounts():
    result = service().group_members("Domain Users")
    assert {(m["name"], m["membership"]) for m in result["members"]} == {
        ("jdoe", "primary"), ("asmith", "primary"), ("disabled", "primary"), ("mnested", "primary")}


def test_group_members_resolve_a_configured_group_dn():
    assert service().group_members("wiki") == {
        "ok": True, "group": "Wiki Readers", "resources": ["wiki"], "members": [], "count": 0,
        "truncated": False, "error": None, "error_code": None}


def test_group_members_are_bounded_and_paged():
    extra = [user(f"bulk{i:03d}", member_of=[dn("VPN-Users")], rid=3000 + i) for i in range(MAX_GROUP_MEMBERS + 10)]
    directory = lab(extra)
    directory.members[dn("VPN-Users").casefold()] += [e.entry_dn.casefold() for e in extra]
    svc = FakeLdapDirectoryService(directory)
    result = svc.group_members("vpn")
    assert result["count"] == MAX_GROUP_MEMBERS and result["truncated"] is True
    pages = [s for s in svc.connections[0].searches if s["filter"].startswith("(memberOf:")]
    assert len(pages) > 1, "large groups must be read page by page"


def test_group_members_unknown_group_and_missing_configured_group():
    missing = service().group_members("No Such Group")
    assert missing["ok"] is False and missing["error_code"] == "not_found"
    unmapped = service(access_groups={"vpn": "Gone-Group"}).group_members("vpn")
    assert unmapped["error_code"] == "not_found" and "Gone-Group" in unmapped["error"]


def test_group_members_refuses_ambiguous_group_names():
    directory = lab()
    twin = group("VPN-Users", 1299, container="OU=Branch")
    directory.entries_by_dn[twin.entry_dn.casefold()] = twin
    result = FakeLdapDirectoryService(directory).group_members("VPN-Users")
    assert result["ok"] is False and result["error_code"] == "ambiguous"


def test_group_members_rejects_empty_or_oversized_names():
    for name in ("", "  ", "x" * 300, None):
        assert service().group_members(name)["error_code"] == "invalid_argument"


# ================================================================
# access_explain
# ================================================================

def reasons(result):
    return [(item["code"], item["suggested_tool"]) for item in result["reasons"]]


def test_access_explain_member_with_healthy_account_has_access():
    result = service().access_explain("jdoe", "VPN")
    assert result["ok"] is True and result["has_access"] is True and result["reasons"] == []
    assert result["checks"] == {"account_exists": True, "enabled": True, "locked": False,
                                "resource_mapped": True, "membership": "direct"}


def test_access_explain_nested_membership_counts():
    result = service().access_explain("mnested", "vpn")
    assert result["has_access"] is True and result["checks"]["membership"] == "nested"


def test_access_explain_locked_member_suggests_unlock_only():
    result = service().access_explain("asmith", "vpn")
    assert result["has_access"] is False
    assert reasons(result) == [("account_locked", "unlock_user")]


def test_access_explain_disabled_non_member_lists_every_reason():
    result = service().access_explain("disabled", "vpn")
    assert result["has_access"] is False
    assert reasons(result) == [("account_disabled", "enable_user"), ("not_a_member", "grant_access")]


def test_access_explain_unknown_account_and_unmapped_resource():
    missing = service().access_explain("nobody", "vpn")
    assert missing["ok"] is True and missing["has_access"] is False
    assert reasons(missing) == [("account_not_found", None)]
    unmapped = service().access_explain("jdoe", "printer")
    assert unmapped["has_access"] is None and reasons(unmapped) == [("resource_not_mapped", None)]


def test_access_explain_reports_a_configured_group_missing_from_the_directory():
    result = service(access_groups={"vpn": "Gone-Group"}).access_explain("jdoe", "vpn")
    assert result["has_access"] is False and reasons(result) == [("resource_group_missing", None)]


def test_access_explain_never_mutates_and_reports_provider_errors():
    svc = service()
    svc.access_explain("disabled", "vpn")
    assert all(connection.unbound for connection in svc.connections)
    failed = service(result_code=51).access_explain("jdoe", "vpn")
    assert failed["ok"] is False and failed["error_code"] == "provider_error" and failed["has_access"] is None


class CappedGroups(MockDirectoryService):

    def user_groups(self, user_id):
        result = super().user_groups(user_id)
        return {**result, "groups": [], "count": 0, "truncated": True}


def test_access_explain_does_not_infer_absence_from_a_capped_group_list():
    result = CappedGroups().access_explain("jdoe", "vpn")
    assert result["ok"] is True and result["has_access"] is None
    assert result["checks"]["membership"] == "unknown"
    assert reasons(result) == [("membership_unknown", None)]


def test_access_explain_rejects_empty_arguments():
    assert service().access_explain(" ", "vpn")["error_code"] == "invalid_argument"


class PartialDirectory(DirectoryService):

    def account_status(self, user_id):
        return {}

    def check_access(self, user_id, resource):
        return {}

    def unlock_user(self, user_id):
        return {}


def test_existing_partial_directory_fakes_still_instantiate():
    with pytest.raises(NotImplementedError):
        PartialDirectory().account_search("jdoe")


# ================================================================
# MockDirectoryService
# ================================================================

def test_mock_reads_follow_mock_access_state():
    directory = MockDirectoryService()
    assert [a["username"] for a in directory.account_search("smith")["accounts"]] == ["asmith"]
    assert directory.user_groups("jdoe")["groups"] == [
        {"name": "VPN-Users", "membership": "direct", "resources": ["vpn"]}]
    assert [m["name"] for m in directory.group_members("admin")["members"]] == ["asmith"]
    assert directory.group_members("vpn access group")["group"] == "VPN-Users"
    assert directory.group_members("printer group")["error_code"] == "not_found"
    directory.access[("jdoe", "admin")] = True
    assert directory.access_explain("jdoe", "admin")["has_access"] is True
    assert reasons(directory.access_explain("asmith", "admin")) == [("account_locked", "unlock_user")]
