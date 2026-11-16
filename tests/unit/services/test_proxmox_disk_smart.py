"""SMART details of one physical disk (direction note plan item B6), read-only.

/nodes/{node}/disks/smart?disk=<devpath> (Sys.Audit) runs smartctl on the node,
so it is one read per disk, asked for by chat only (never the dashboard's
minute refresh). The device must be one /nodes/{node}/disks/list reports.
Only a fixed set of numbers leaves the service: temperature, power-on hours,
power cycles, unsafe shutdowns, media errors, reallocated/pending/offline
uncorrectable sectors, spare capacity and its threshold, wear. Serial numbers,
WWNs, vendor-specific rows and the raw smartctl text never do.
"""
import json

import httpx
import pytest

from services.proxmox import ProxmoxConfig, ProxmoxService

TOKEN_ID = "reporter@pam!readonly"
TOKEN_SECRET = "fixture-token-secret"
LIST = [{"devpath": "/dev/nvme0n1", "type": "nvme", "model": "INTEL SSDPEKNW512G8", "size": 512110190592,
         "health": "PASSED", "wearout": 88, "serial": "BTNH0000000X512A"},
        {"devpath": "/dev/sda", "type": "hdd", "model": "WDC WD40EFRX", "size": 4000787030016, "health": "PASSED",
         "wearout": "N/A", "serial": "WD-WCC0000000"}]
# The live hummingbird NVMe answer (PVE 9.2), serial-free as Proxmox returns it.
NVME_TEXT = """Critical Warning:                   0x00
Temperature:                        30 Celsius
Available Spare:                    100%
Available Spare Threshold:          10%
Percentage Used:                    12%
Data Units Read:                    150,575,344 [77.0 TB]
Data Units Written:                 69,353,843 [35.5 TB]
Power Cycles:                       2,712
Power On Hours:                     7,686
Unsafe Shutdowns:                   95
Media and Data Integrity Errors:    0
Error Information Log Entries:      158
Serial Number:                      BTNH0000000X512A
"""
ATA = [{"id": "  5", "name": "Reallocated_Sector_Ct", "flags": "PO--CK", "value": 200, "normalized": 200,
        "worst": 200, "threshold": 140, "fail": "-", "raw": "8"},
       {"id": "  9", "name": "Power_On_Hours", "flags": "-O--CK", "value": 41, "normalized": 41, "worst": 41,
        "threshold": 0, "fail": "-", "raw": "43215"},
       {"id": " 12", "name": "Power_Cycle_Count", "flags": "-O--CK", "value": 100, "normalized": 100,
        "worst": 100, "threshold": 0, "fail": "-", "raw": "61"},
       {"id": "194", "name": "Temperature_Celsius", "flags": "-O---K", "value": 116, "normalized": 116,
        "worst": 103, "threshold": 0, "fail": "-", "raw": "34 (Min/Max 18/47)"},
       {"id": "197", "name": "Current_Pending_Sector", "flags": "-O--CK", "value": 200, "normalized": 200,
        "worst": 200, "threshold": 0, "fail": "-", "raw": "2"},
       {"id": "198", "name": "Offline_Uncorrectable", "flags": "----CK", "value": 100, "normalized": 100,
        "worst": 253, "threshold": 0, "fail": "-", "raw": "0"},
       {"id": "240", "name": "Head_Flying_Hours", "flags": "------", "value": 100, "normalized": 100,
        "worst": 253, "threshold": 0, "fail": "-", "raw": "vendor 0x0000"}]


def read(smart_answers, node="pve-a", disk="/dev/nvme0n1", calls=None):
    def handler(request):
        if calls is not None:
            calls.append((request.url.path, dict(request.url.params)))
        if request.url.path.endswith("/disks/list"):
            return httpx.Response(200, json={"data": LIST}, request=request)
        status, data = smart_answers[request.url.params["disk"]]
        return httpx.Response(status, json={"data": data}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        return ProxmoxService(ProxmoxConfig("https://pve.example.test:8006", TOKEN_ID, TOKEN_SECRET),
                              client).disk_smart(node, disk)
    finally:
        client.close()


def test_an_nvme_disk_reads_its_fixed_numbers_from_the_smart_text():
    calls = []
    result = read({"/dev/nvme0n1": (200, {"type": "text", "health": "PASSED", "wearout": 88, "text": NVME_TEXT})},
                  calls=calls)
    assert calls == [("/api2/json/nodes/pve-a/disks/list", {}),
                     ("/api2/json/nodes/pve-a/disks/smart", {"disk": "/dev/nvme0n1"})]
    data = result["data"]
    assert data["schema"] == "proxmox-disk-smart.v1"
    assert (data["node"], data["device"], data["type"], data["model"]) == (
        "pve-a", "/dev/nvme0n1", "nvme", "INTEL SSDPEKNW512G8")
    assert data["health"] == "passed" and data["wear_used_percent"] == 12
    assert data["measures"] == {"temperature_celsius": 30, "power_on_hours": 7686, "power_cycles": 2712,
                                "unsafe_shutdowns": 95, "media_errors": 0, "available_spare_percent": 100,
                                "available_spare_threshold_percent": 10, "reallocated_sectors": None,
                                "pending_sectors": None, "offline_uncorrectable": None}
    assert data["issues"] == []
    text = json.dumps(result)
    for leaked in ("BTNH", "Serial", "Data Units", "Error Information", "text"):
        assert leaked not in text, leaked


def test_an_ata_disk_reads_its_fixed_attributes_and_flags_bad_sectors():
    data = read({"/dev/sda": (200, {"type": "ata", "health": "PASSED", "attributes": ATA})}, disk="/dev/sda")["data"]
    assert data["measures"] == {"temperature_celsius": 34, "power_on_hours": 43215, "power_cycles": 61,
                                "unsafe_shutdowns": None, "media_errors": None, "available_spare_percent": None,
                                "available_spare_threshold_percent": None, "reallocated_sectors": 8,
                                "pending_sectors": 2, "offline_uncorrectable": 0}
    # Sectors the disk had to move, or cannot read yet, mean it is failing even while SMART says PASSED.
    assert data["issues"] == ["reallocated_sectors", "pending_sectors"]
    assert "vendor" not in json.dumps(data) and "Head_Flying" not in json.dumps(data)


def test_spare_capacity_below_its_threshold_and_media_errors_are_issues():
    text = NVME_TEXT.replace("Available Spare:                    100%", "Available Spare:                    8%")
    text = text.replace("Media and Data Integrity Errors:    0", "Media and Data Integrity Errors:    3")
    data = read({"/dev/nvme0n1": (200, {"type": "text", "health": "PASSED", "text": text})})["data"]
    assert data["issues"] == ["media_errors", "spare_below_threshold"]


def test_a_bare_device_name_reads_the_same_disk():
    # Users say "nvme0n1" or "sda"; the grounded value gets the /dev/ prefix Proxmox lists.
    calls = []
    data = read({"/dev/nvme0n1": (200, {"type": "text", "health": "PASSED", "text": NVME_TEXT})},
                disk="nvme0n1", calls=calls)["data"]
    assert data["device"] == "/dev/nvme0n1"
    assert calls[-1] == ("/api2/json/nodes/pve-a/disks/smart", {"disk": "/dev/nvme0n1"})


def test_a_disk_proxmox_does_not_list_on_that_node_is_never_read():
    calls = []
    with pytest.raises(ValueError, match="not a disk Proxmox lists"):
        read({}, disk="/dev/sdz", calls=calls)
    assert [path for path, _ in calls] == ["/api2/json/nodes/pve-a/disks/list"]


@pytest.mark.parametrize("node,disk", [("../x", "/dev/sda"), ("pve-a", "../sda"), ("pve-a", "/dev/../etc/passwd"),
                                       ("pve-a", "/dev/sda;rm"), ("pve-a", "/sys/block/sda"), ("pve-a", 5)])
def test_only_an_exact_node_and_device_path_are_accepted(node, disk):
    calls = []
    with pytest.raises(ValueError):
        read({}, node=node, disk=disk, calls=calls)
    assert calls == []


def test_smart_proxmox_could_not_run_is_unknown_not_healthy():
    data = read({"/dev/nvme0n1": (200, {"health": "UNKNOWN"})})["data"]
    assert data["health"] == "unknown" and data["measures"]["temperature_celsius"] is None


@pytest.mark.parametrize("answer", [
    {"type": "ata", "health": "PASSED", "attributes": "5"},
    {"type": "ata", "health": "PASSED", "attributes": [{"name": "Reallocated_Sector_Ct", "raw": 8}]},
    {"type": "text", "health": "PASSED", "text": 7},
    [],
])
def test_a_malformed_smart_answer_is_refused(answer):
    with pytest.raises(RuntimeError, match="invalid answer"):
        read({"/dev/nvme0n1": (200, answer)})
