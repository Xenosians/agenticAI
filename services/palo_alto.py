from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import httpx


SYSTEM_INFO_COMMAND = "<show><system><info></info></system></show>"


@dataclass(frozen=True)
class PaloAltoConfig:
    base_url: str
    api_key: str
    verify_tls: bool = True
    timeout_seconds: float = 10.0
    allow_insecure_http: bool = False

    def normalized_base_url(self) -> str:
        value = self.base_url.strip().rstrip("/")
        parsed = urlsplit(value)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValueError("Palo Alto base URL must be an absolute HTTP(S) URL.")
        if parsed.scheme != "https" and not self.allow_insecure_http:
            raise ValueError(
                "Palo Alto API requires HTTPS unless PALO_ALTO_ALLOW_INSECURE_HTTP=true is explicitly set for a lab."
            )
        return value


class PaloAltoService:
    """Read-only PAN-OS XML API adapter for first E2E integration."""

    def __init__(self, config: PaloAltoConfig, client: httpx.Client | None = None) -> None:
        self.config = config
        self.base_url = config.normalized_base_url()
        self.client = client or httpx.Client(
            verify=config.verify_tls,
            timeout=config.timeout_seconds,
            headers={"X-PAN-KEY": config.api_key},
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def _op(self, command_xml: str) -> ET.Element:
        response = self.client.post(
            f"{self.base_url}/api",
            params={
                "type": "op",
                "cmd": command_xml,
            },
            headers={
                "X-PAN-KEY":
                    self.config.api_key,
            },
        )
        response.raise_for_status()

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            raise RuntimeError("PAN-OS returned invalid XML.") from exc

        status = (root.attrib.get("status") or "").strip().lower()
        if status != "success":
            message = " ".join(
                text.strip()
                for text in root.itertext()
                if isinstance(text, str) and text.strip()
            )
            raise RuntimeError(f"PAN-OS XML API rejected the request: {message[:300]}")

        return root

    @staticmethod
    def _text(root: ET.Element, path: str) -> str | None:
        node = root.find(path)
        if node is None or node.text is None:
            return None
        value = node.text.strip()
        return value or None

    def system_info(self) -> dict:
        root = self._op(SYSTEM_INFO_COMMAND)
        result = root.find("./result")
        if result is None:
            raise RuntimeError("PAN-OS system-info response has no result node.")

        return {
            "ok": True,
            "status": "success",
            "hostname": self._text(result, "hostname"),
            "ip_address": self._text(result, "ip-address"),
            "model": self._text(result, "model"),
            "serial": self._text(result, "serial"),
            "sw_version": self._text(result, "sw-version"),
            "app_version": self._text(result, "app-version"),
            "av_version": self._text(result, "av-version"),
            "uptime": self._text(result, "uptime"),
        }


def _setting(settings, name: str, default=None):
    return getattr(settings, name, default)


def palo_alto_configured(settings) -> bool:
    return bool(
        isinstance(_setting(settings, "palo_alto_base_url"), str)
        and _setting(settings, "palo_alto_base_url").strip()
        and isinstance(_setting(settings, "palo_alto_api_key"), str)
        and _setting(settings, "palo_alto_api_key").strip()
    )


def palo_alto_integration_status(settings) -> dict:
    base_url = _setting(settings, "palo_alto_base_url")
    host = None
    if isinstance(base_url, str) and base_url.strip():
        parsed = urlsplit(base_url.strip())
        host = parsed.hostname

    return {
        "id": "palo_alto",
        "name": "Palo Alto Networks PAN-OS",
        "configured": palo_alto_configured(settings),
        "host": host,
        "verify_tls": bool(_setting(settings, "palo_alto_verify_tls", True)),
        "mode": "read_only_v1",
    }


def build_palo_alto_service(settings) -> PaloAltoService | None:
    if not palo_alto_configured(settings):
        return None

    return PaloAltoService(
        PaloAltoConfig(
            base_url=_setting(settings, "palo_alto_base_url"),
            api_key=_setting(settings, "palo_alto_api_key"),
            verify_tls=bool(_setting(settings, "palo_alto_verify_tls", True)),
            timeout_seconds=float(_setting(settings, "palo_alto_timeout_seconds", 10.0)),
            allow_insecure_http=bool(_setting(settings, "palo_alto_allow_insecure_http", False)),
        )
    )
