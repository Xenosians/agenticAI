import httpx
import pytest

from services.palo_alto import PaloAltoConfig, PaloAltoService


def test_system_info_uses_pan_key_header_and_parses_xml():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["header"] = request.headers.get("X-PAN-KEY")
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            text=(
                '<response status="success"><result><system>'
                '<hostname>fw-lab</hostname>'
                '<model>PA-VM</model>'
                '<serial>001122</serial>'
                '<sw-version>11.1.4</sw-version>'
                '<uptime>1 day</uptime>'
                '</system></result></response>'
            ),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    service = PaloAltoService(
        PaloAltoConfig(
            base_url="https://firewall.local",
            api_key="secret-test-key",
        ),
        client=client,
    )

    result = service.system_info()

    assert seen["header"] == "secret-test-key"
    assert "type=op" in seen["url"]
    assert result["hostname"] == "fw-lab"
    assert result["model"] == "PA-VM"
    assert result["sw_version"] == "11.1.4"


@pytest.mark.parametrize("xml, message", [
    ('<response status="success"><result/></response>', "result/system"),
    ('<response status="success"><result><system/></result></response>', "required fields"),
    ('<response status="error"><msg>Invalid credentials</msg></response>', "rejected"),
    ('<html>proxy failure', "invalid XML"),
])
def test_system_info_rejects_invalid_or_incomplete_provider_responses(xml, message):
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=xml)
    )) as client:
        service = PaloAltoService(
            PaloAltoConfig(base_url="https://firewall.local", api_key="test-key"), client=client
        )
        with pytest.raises(RuntimeError, match=message):
            service.system_info()


def test_system_info_propagates_provider_http_failure():
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(503, text="Unavailable")
    )) as client:
        service = PaloAltoService(
            PaloAltoConfig(base_url="https://firewall.local", api_key="test-key"), client=client
        )
        with pytest.raises(httpx.HTTPStatusError):
            service.system_info()
