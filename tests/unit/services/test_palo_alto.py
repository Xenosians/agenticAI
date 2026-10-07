import httpx

from services.palo_alto import PaloAltoConfig, PaloAltoService


def test_system_info_uses_pan_key_header_and_parses_xml():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["header"] = request.headers.get("X-PAN-KEY")
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            text=(
                '<response status="success"><result>'
                '<hostname>fw-lab</hostname>'
                '<model>PA-VM</model>'
                '<serial>001122</serial>'
                '<sw-version>11.1.4</sw-version>'
                '<uptime>1 day</uptime>'
                '</result></response>'
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
