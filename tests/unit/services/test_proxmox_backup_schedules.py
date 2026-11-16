"""Backups judged against each guest's own backup job (direction note plan item B4).

A fixed 8 days flags a monthly job's guests as overdue for three weeks of
every month and misses a daily job that stopped yesterday. Proxmox computes
a job's calendar events itself (/cluster/jobs/schedule-analyze, any user) and
the guests a job includes (/cluster/backup/{id}/included_volumes, Sys.Audit):
nothing is parsed here. A guest in an enabled job whose last scheduled run is
at least 12 hours old (a run can take hours) has missed that run when its
newest copy is older; the 8-day age rule stays for guests without such a job.
"""
import time

import httpx
import pytest

from services.proxmox import ProxmoxConfig, ProxmoxService
from tests.unit.services.test_proxmox_hardware import (BACKUP_STORAGES, DAY, NOW, RESOURCES, TOKEN_ID,
                                                       TOKEN_SECRET, volume)

HOUR = 3600
ANALYZE = "/api2/json/cluster/jobs/schedule-analyze"


def schedule_events(period: int, first: int):
    """Proxmox's answer for a fixed-period schedule: the next events after starttime."""
    def answer(params):
        start, count = int(params["starttime"]), int(params.get("iterations", 10))
        step = (start - first) // period + 1  # floor division: events exist before `first` too
        return [{"timestamp": first + (step + index) * period, "utc": "x"} for index in range(count)]
    return answer


def routes(*, jobs, included, schedules, local=(), resources=RESOURCES, uncovered=()):
    table = {"/api2/json/cluster/resources": (200, resources),
             "/api2/json/cluster/backup": (200, jobs),
             "/api2/json/cluster/backup-info/not-backed-up": (200, [{"vmid": vmid, "type": "lxc"}
                                                                    for vmid in uncovered]),
             "/api2/json/nodes/pve-a/storage": (200, BACKUP_STORAGES[:1]),
             "/api2/json/nodes/pve-a/storage/local/content": (200, list(local)),
             ANALYZE: schedules}
    for job, guests in included.items():
        table[f"/api2/json/cluster/backup/{job}/included_volumes"] = (
            guests if isinstance(guests, tuple) else
            (200, {"children": [{"id": vmid, "type": "qemu", "name": f"guest-{vmid}"} for vmid in guests]}))
    return table


def read(table, calls=None):
    def handler(request):
        if calls is not None:
            calls.append((request.url.path, dict(request.url.params)))
        entry = table[request.url.path]
        if callable(entry):
            return httpx.Response(200, json={"data": entry(dict(request.url.params))}, request=request)
        status, data = entry if isinstance(entry, tuple) else entry[dict(request.url.params)["schedule"]]
        return httpx.Response(status, json={"data": data}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        return ProxmoxService(ProxmoxConfig("https://pve.example.test:8006", TOKEN_ID, TOKEN_SECRET),
                              client).backups()["data"]
    finally:
        client.close()


def by_schedule(**answers):
    return lambda params: answers[params["schedule"]](params)


@pytest.fixture(autouse=True)
def frozen(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: NOW)


DAILY = schedule_events(DAY, NOW - 20 * HOUR)  # due 20 h ago, next in 4 h


def test_a_guest_the_last_scheduled_run_left_without_a_copy_has_missed_that_run():
    data = read(routes(jobs=[{"id": "nightly", "enabled": 1, "schedule": "02:00"}],
                       included={"nightly": [101, 102]}, schedules=by_schedule(**{"02:00": DAILY}),
                       local=[volume("local", 101, 0.5), volume("local", 102, 2)], uncovered=[103]))
    rows = {row["vmid"]: row for row in data["guests"]}
    assert rows[101]["last_scheduled_run"] == NOW - 20 * HOUR and rows[101]["issues"] == []
    # Two days old: within the fixed 8 days, but last night's run left no copy.
    assert rows[102]["issues"] == ["missed_run"]
    assert rows[103]["last_scheduled_run"] is None
    assert data["summary"]["missed_run"] == 1 and data["summary"]["stale"] == 0
    assert data["missed_run_grace_hours"] == 12


def test_a_monthly_job_keeps_a_three_week_old_copy_from_being_called_overdue():
    monthly = schedule_events(30 * DAY, NOW - 20 * DAY)
    data = read(routes(jobs=[{"id": "monthly", "enabled": 1, "schedule": "monthly"}],
                       included={"monthly": [101, 102]}, schedules=by_schedule(monthly=monthly),
                       local=[volume("local", 101, 19), volume("local", 102, 21)]))
    rows = {row["vmid"]: row for row in data["guests"]}
    assert rows[101]["last_scheduled_run"] == NOW - 20 * DAY
    assert rows[101]["issues"] == []  # 19 days old, yet newer than the run due 20 days ago
    assert rows[102]["issues"] == ["missed_run"]


def test_a_run_due_less_than_twelve_hours_ago_is_not_judged_yet():
    recent = schedule_events(DAY, NOW - 2 * HOUR)  # tonight's run started 2 h ago
    data = read(routes(jobs=[{"id": "nightly", "enabled": 1, "schedule": "02:00"}],
                       included={"nightly": [101]}, schedules=by_schedule(**{"02:00": recent}),
                       local=[volume("local", 101, 1.05)]))
    row = next(row for row in data["guests"] if row["vmid"] == 101)
    assert row["last_scheduled_run"] == NOW - 26 * HOUR and row["issues"] == []


def test_a_frequent_schedule_still_finds_its_latest_due_run():
    quarter = schedule_events(15 * 60, NOW - 7 * 60)
    data = read(routes(jobs=[{"id": "often", "enabled": 1, "schedule": "*/15"}],
                       included={"often": [101]}, schedules=by_schedule(**{"*/15": quarter}),
                       local=[volume("local", 101, 0.6)]))
    row = next(row for row in data["guests"] if row["vmid"] == 101)
    # The newest event at least 12 hours old, not merely the first page of events.
    assert NOW - 12 * HOUR - 15 * 60 < row["last_scheduled_run"] <= NOW - 12 * HOUR
    assert row["issues"] == ["missed_run"]


@pytest.mark.parametrize("change", [
    {"jobs": [{"id": "nightly", "enabled": 0, "schedule": "02:00"}]},
    {"jobs": [{"id": "nightly", "enabled": 1}]},
    {"included": {"nightly": (403, None)}},
])
def test_without_a_readable_enabled_scheduled_job_the_age_rule_applies(change):
    options = dict(jobs=[{"id": "nightly", "enabled": 1, "schedule": "02:00"}], included={"nightly": [102]},
                   schedules=by_schedule(**{"02:00": DAILY}), local=[volume("local", 102, 12)])
    options.update(change)
    data = read(routes(**options))
    row = next(row for row in data["guests"] if row["vmid"] == 102)
    assert row["last_scheduled_run"] is None and row["issues"] == ["backup_stale"]


def test_a_schedule_proxmox_cannot_analyze_falls_back_to_the_age_rule():
    data = read(routes(jobs=[{"id": "odd", "enabled": 1, "schedule": "02:00"}], included={"odd": [102]},
                       schedules={"02:00": (400, None)}, local=[volume("local", 102, 12)]))
    row = next(row for row in data["guests"] if row["vmid"] == 102)
    assert row["last_scheduled_run"] is None and row["issues"] == ["backup_stale"]


def test_a_guest_in_two_jobs_is_judged_by_the_most_recent_due_run():
    weekly = schedule_events(7 * DAY, NOW - 3 * DAY)
    data = read(routes(jobs=[{"id": "nightly", "enabled": 1, "schedule": "02:00"},
                             {"id": "weekly", "enabled": 1, "schedule": "sat 03:00"}],
                       included={"nightly": [101], "weekly": [101]},
                       schedules=by_schedule(**{"02:00": DAILY, "sat 03:00": weekly}),
                       local=[volume("local", 101, 2)]))
    row = next(row for row in data["guests"] if row["vmid"] == 101)
    assert row["last_scheduled_run"] == NOW - 20 * HOUR and row["issues"] == ["missed_run"]


def test_schedule_reads_are_bounded_and_shared_between_jobs():
    calls = []
    read(routes(jobs=[{"id": "a", "enabled": 1, "schedule": "02:00"}, {"id": "b", "enabled": 1, "schedule": "02:00"}],
                included={"a": [101], "b": [102]}, schedules=by_schedule(**{"02:00": DAILY}),
                local=[volume("local", 101, 1), volume("local", 102, 1)]), calls)
    analyses = [params for path, params in calls if path == ANALYZE]
    assert len(analyses) == 1 and analyses[0]["schedule"] == "02:00" and analyses[0]["iterations"] == "100"
    assert sum(path.endswith("/included_volumes") for path, _ in calls) == 2


@pytest.mark.parametrize("answer", [
    (200, {"children": [{"id": "101", "type": "qemu"}]}),
    (200, {"children": [{"id": 0, "type": "qemu"}]}),
    (200, {"children": "101"}),
    (200, [101]),
])
def test_a_malformed_included_guests_answer_is_refused(answer):
    with pytest.raises(RuntimeError, match="invalid answer"):
        read(routes(jobs=[{"id": "nightly", "enabled": 1, "schedule": "02:00"}], included={"nightly": answer},
                    schedules=by_schedule(**{"02:00": DAILY}), local=[volume("local", 101, 1)]))


@pytest.mark.parametrize("events", [[{"timestamp": "soon"}], [{"timestamp": -1}], {"timestamp": 1}])
def test_a_malformed_schedule_answer_is_refused(events):
    with pytest.raises(RuntimeError, match="invalid answer"):
        read(routes(jobs=[{"id": "nightly", "enabled": 1, "schedule": "02:00"}], included={"nightly": [101]},
                    schedules={"02:00": (200, events)}, local=[volume("local", 101, 1)]))
