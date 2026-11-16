"""One Proxmox node's load over a recent window, read-only (direction note H1).

Two reads of /nodes/{node}/rrddata (Sys.Audit): the AVERAGE series for typical
values and the MAX series for peaks (at the one-minute resolution of the hour
and day windows they are the same; for a week or more MAX keeps short peaks).
Proxmox fills minutes it has no sample for with null; those are gaps, never
zero. CPU and IO wait arrive as fractions of all cores, PSI pressure as
percent of time. Thresholds: peak CPU 90 %, lowest available memory under
10 % of the total, any full memory stall, full IO stall at 10 % or more and
swap use at 10 % or more of the swap space.
"""
import httpx
import pytest

from services.proxmox import ProxmoxConfig, ProxmoxService

GIB = 2 ** 30


def sample(time, **values):
    row = {"time": time, "maxcpu": 8, "memtotal": 6 * GIB, "swaptotal": 6 * GIB, "roottotal": 100 * GIB,
           "cpu": 0.01, "iowait": 0.001, "loadavg": 0.1, "memused": 2 * GIB, "memavailable": 4 * GIB,
           "swapused": 0, "pressurecpusome": 0, "pressureiosome": 0, "pressureiofull": 0, "pressurememorysome": 0,
           "pressurememoryfull": 0, "netin": 1000.0, "netout": 500.0, "rootused": 10 * GIB}
    row.update(values)
    return row


def gap(time):
    return {"time": time}


AVERAGE = [sample(0, cpu=0.01), sample(60, cpu=0.03, memused=3 * GIB, memavailable=3 * GIB), gap(120),
           sample(180, cpu=0.02, netin=3000.0, rootused=11 * GIB)]
PEAK = [sample(0, cpu=0.02), sample(60, cpu=0.06, loadavg=0.9, memused=3 * GIB, pressureiofull=2.5),
        gap(120), sample(180, cpu=0.02, netin=9000.0, rootused=11 * GIB)]


def read(average=AVERAGE, peak=PEAK, timeframe="day", node="pve-a", calls=None):
    def handler(request):
        params = dict(request.url.params)
        if calls is not None:
            calls.append((request.url.path, params))
        rows = average if params.get("cf") == "AVERAGE" else peak
        return httpx.Response(200, json={"data": rows}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    service = ProxmoxService(ProxmoxConfig("https://pve.example.test:8006", "reporter@pam!readonly", "secret"), client)
    try:
        return service.node_load(node, timeframe)["data"]
    finally:
        client.close()


def test_average_and_peak_load_come_from_two_series_and_gaps_are_not_zero():
    calls = []
    data = read(calls=calls)
    assert calls == [("/api2/json/nodes/pve-a/rrddata", {"timeframe": "day", "cf": "AVERAGE"}),
                     ("/api2/json/nodes/pve-a/rrddata", {"timeframe": "day", "cf": "MAX"})]
    assert data == {
        "schema": "proxmox-node-load.v1", "node": "pve-a", "timeframe": "day", "step_seconds": 60,
        "samples": 3, "expected_samples": 4, "from": "1970-01-01T00:00:00+00:00", "to": "1970-01-01T00:03:00+00:00",
        "cpu": {"cores": 8, "average_percent": 2.0, "peak_percent": 6.0},
        "io_wait": {"average_percent": 0.1, "peak_percent": 0.1},
        "load": {"average": 0.1, "peak": 0.9},
        "memory": {"total_bytes": 6 * GIB, "average_used_bytes": 7 * GIB / 3, "peak_used_bytes": 3 * GIB,
                   "lowest_available_bytes": 3 * GIB},
        "swap": {"total_bytes": 6 * GIB, "peak_used_bytes": 0},
        "pressure": {"cpu_some": 0, "io_some": 0, "io_full": 2.5, "memory_some": 0, "memory_full": 0},
        "network": {"average_in_bytes_per_second": 5000 / 3, "peak_in_bytes_per_second": 9000.0,
                    "average_out_bytes_per_second": 500.0, "peak_out_bytes_per_second": 500.0},
        "root_disk": {"used_bytes": 11 * GIB, "total_bytes": 100 * GIB},
        "issues": [],
    }


def test_busy_low_memory_stalls_and_swapping_are_named_most_urgent_first():
    peak = [sample(0, cpu=0.95, memavailable=0.5 * GIB, pressurememoryfull=0.4, pressureiofull=12.0,
                   swapused=GIB)]
    data = read(average=[sample(0, memavailable=0.5 * GIB)], peak=peak)
    assert data["issues"] == ["memory_low", "memory_stalls", "io_stalls", "cpu_busy", "swapping"]
    # Each threshold is a boundary: just below it is not an issue.
    calm = [sample(0, cpu=0.899, memavailable=0.6 * GIB + 1, pressureiofull=9.99, swapused=0.6 * GIB - 1)]
    assert read(average=[sample(0, memavailable=0.6 * GIB + 1)], peak=calm)["issues"] == []


def test_load_and_stall_values_are_rounded_as_proxmox_sends_long_fractions():
    # Live, Proxmox sent loadavg 0.516333333333333 and pressureiosome 6.40266666666667.
    peak = [sample(0, loadavg=0.516333333333333, pressurecpusome=0.0433333333333333,
                   pressureiosome=6.40266666666667, pressureiofull=6.29550000000001)]
    data = read(average=[sample(0, loadavg=0.0700694444444444)], peak=peak)
    assert data["load"] == {"average": 0.07, "peak": 0.52}
    assert data["pressure"] == {"cpu_some": 0.04, "io_some": 6.4, "io_full": 6.3, "memory_some": 0,
                                "memory_full": 0}


def test_a_window_without_any_sample_says_so_instead_of_reporting_zero_load():
    data = read(average=[gap(0), gap(60)], peak=[gap(0), gap(60)])
    assert (data["samples"], data["expected_samples"]) == (0, 2)
    assert data["cpu"] is None and data["memory"] is None and data["issues"] == []


@pytest.mark.parametrize("timeframe", ["hour", "day", "week", "month"])
def test_each_supported_window_is_read_as_named(timeframe):
    calls = []
    read(timeframe=timeframe, calls=calls)
    assert {params["timeframe"] for _, params in calls} == {timeframe}


@pytest.mark.parametrize("node,timeframe", [("../pve-a", "day"), ("pve-a", "year"), ("pve-a", "DAY"), ("pve-a", None)])
def test_the_node_and_window_must_be_exact(node, timeframe):
    with pytest.raises(ValueError):
        read(node=node, timeframe=timeframe)


@pytest.mark.parametrize("bad", [[{"cpu": 0.1}], [sample(0, cpu="high")], [sample(0, cpu=-0.1)], {"time": 0}])
def test_malformed_series_are_refused(bad):
    with pytest.raises(RuntimeError, match="invalid answer"):
        read(average=bad)
