import json
import httpx
import pytest
from services.openwrt import OpenWrtConfig, OpenWrtService


def service(handler):
    return OpenWrtService(OpenWrtConfig("https://router.example", "reader", "test-secret"),
                          httpx.Client(transport=httpx.MockTransport(handler)))


def test_read_uses_scoped_login_and_exact_interface_without_credentials_in_result():
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body["params"])
        data = {"ubus_rpc_session": "a" * 32} if len(calls) == 1 else {"up": True}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": [0, data]})
    s = service(handler)
    result = s.interface_status("guest")
    assert result["data"] == {"up": True}
    assert calls[1] == ["a" * 32, "network.interface.guest", "status", {}]
    assert "test-secret" not in json.dumps(result)


@pytest.mark.parametrize("interface", ["lan;reboot", "../wan", "*", "", "lan.status"])
def test_invalid_interface_is_rejected_before_network(interface):
    def handler(request):
        pytest.fail("Invalid input must not reach provider")
    with pytest.raises(ValueError):
        service(handler).interface_status(interface)


@pytest.mark.parametrize("payload", [
    {"id": 1, "result": [6]}, {"id": 2, "result": [0, {}]},
    {"id": 1, "error": {"code": -32000}}, [],
])
def test_rpc_failures_are_not_success(payload):
    with pytest.raises(RuntimeError):
        service(lambda r: httpx.Response(200, json=payload)).system_info()


@pytest.mark.parametrize("url", ["http://router.example", "https://user:pass@router.example", "https://router.example/path", "http://localhost"])
def test_insecure_or_credential_urls_rejected(url):
    with pytest.raises(ValueError):
        OpenWrtConfig(url, "reader", "secret", allow_loopback_http=True).origin()


def test_explicit_loopback_lab_transport():
    assert OpenWrtConfig("http://127.0.0.1:18080", "reader", "secret", allow_loopback_http=True).origin() == "http://127.0.0.1:18080"


def test_oversized_response_stops_at_transport_boundary():
    with pytest.raises(RuntimeError, match="bounded response size"):
        service(lambda r: httpx.Response(200, content=b" " * 1_100_000)).system_info()
