from copy import deepcopy
from types import SimpleNamespace
import asyncio
from contextlib import asynccontextmanager, contextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from agent.requested_workflow import RequestedWorkflowContract
from api import app as api
from subagents.core.definitions.types import HubResult


def contract():
    return RequestedWorkflowContract.model_validate({"version": 1, "steps": [{
        "id": "projects", "capability": "jira_project_list", "grounded_arguments": {},
        "after_success_of": None,
    }]})


def payload():
    reviewed = contract()
    return {
        "job_id": "isolated-job", "attempt": 1, "user_id": "owner-one",
        "message": "List available Jira projects", "context": [],
        "workflow_contract": reviewed.model_dump(), "workflow_contract_hash": reviewed.content_hash(),
    }


@pytest.mark.parametrize("change", ["missing_hash", "missing_contract", "wrong_hash", "changed_contract"])
def test_internal_request_rejects_unbound_contract(change):
    value = payload()
    if change == "missing_hash":
        value.pop("workflow_contract_hash")
    elif change == "missing_contract":
        value.pop("workflow_contract")
    elif change == "wrong_hash":
        value["workflow_contract_hash"] = "0" * 64
    else:
        value["workflow_contract"]["steps"][0]["capability"] = "ticket_get"
    with pytest.raises(ValidationError):
        api.JobExecuteRequest.model_validate(value)


def test_internal_request_preserves_exact_pair_and_legacy_absence():
    value = payload()
    assert api.JobExecuteRequest.model_validate(value).model_dump() == value
    value.pop("workflow_contract")
    value.pop("workflow_contract_hash")
    parsed = api.JobExecuteRequest.model_validate(value)
    assert parsed.workflow_contract is None and parsed.workflow_contract_hash is None


def test_authenticated_http_admission_checks_binding_before_scheduling(monkeypatch):
    from api import extensions
    scheduled = []
    monkeypatch.setattr(extensions, "get_settings", lambda: SimpleNamespace(
        require_internal_job_token=lambda: "isolated-internal-token",
    ))
    monkeypatch.setattr(api, "require_ready_runtime", lambda request: SimpleNamespace(active_jobs={}))
    monkeypatch.setattr(api, "schedule_job", lambda runtime, request: scheduled.append(request))
    application = FastAPI()
    application.add_api_route("/execute", api.execute_durable_job, methods=["POST"], status_code=202)
    client = TestClient(application)
    body = payload()
    assert client.post("/execute", json=body).status_code == 401
    assert not scheduled
    headers = {"x-internal-token": "isolated-internal-token"}
    wrong = deepcopy(body)
    wrong["workflow_contract_hash"] = "1" * 64
    assert client.post("/execute", json=wrong, headers=headers).status_code == 422
    assert not scheduled
    response = client.post("/execute", json=body, headers=headers)
    assert response.status_code == 202
    assert len(scheduled) == 1
    assert scheduled[0].workflow_contract.model_dump() == body["workflow_contract"]
    assert scheduled[0].workflow_contract_hash == body["workflow_contract_hash"]


@pytest.mark.parametrize("reviewed", [False, True])
def test_background_job_passes_bound_scope_to_hub_without_changing_outcome(monkeypatch, reviewed):
    captured, completions = [], []

    class Hub:
        async def run(self, message, **kwargs):
            captured.append((message, kwargs))
            return HubResult(status="error", user_request=message, error="isolated refusal", answer="isolated refusal")

    @contextmanager
    def owner(*args, **kwargs):
        yield None, False

    @asynccontextmanager
    async def progress(*args, **kwargs):
        yield

    async def heartbeat(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr("services.extensions.context.owner_execution", owner)
    monkeypatch.setattr("services.job_progress.job_progress", progress)
    monkeypatch.setattr(api, "job_heartbeat_worker", heartbeat)
    body = payload()
    if not reviewed:
        body.pop("workflow_contract")
        body.pop("workflow_contract_hash")
    request = api.JobExecuteRequest.model_validate(body)
    runtime = SimpleNamespace(
        settings=SimpleNamespace(), hub=Hub(), active_jobs={request.job_id: object()},
        trajectory_recorder=SimpleNamespace(record=lambda **kwargs: None),
        completion_outbox=SimpleNamespace(put=lambda *args: completions.append(args)),
        outbox_wakeup=asyncio.Event(),
    )
    asyncio.run(api.execute_job(runtime, request))
    assert len(captured) == 1
    message, options = captured[0]
    assert message == request.message
    assert options["context"] == []
    if reviewed:
        assert options["workflow_contract"] == request.workflow_contract
    else:
        assert "workflow_contract" not in options
    assert len(completions) == 1
    assert completions[0][2]["status"] == "failed"
    assert request.job_id not in runtime.active_jobs
