"""Bounded transports. Provider responses are data, never execution authority."""
import json
import os
from urllib.parse import urlsplit

import httpx


DEFAULT_ORIGINS = {"https://openrouter.ai", "https://opencode.ai", "https://api.groq.com", "https://generativelanguage.googleapis.com",
                   "http://127.0.0.1:11434", "http://127.0.0.1:1234"}


def validate_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.username or parts.password or parts.query or parts.fragment or not parts.hostname:
        raise ValueError("Use an endpoint URL without credentials, query parameters or fragments")
    origins = DEFAULT_ORIGINS | set(filter(None, os.environ.get("EXTENSION_ALLOWED_ORIGINS", "").split(",")))
    origin = f"{parts.scheme}://{parts.netloc}"
    if origin not in origins:
        raise ValueError("Endpoint origin is not in the server's EXTENSION_ALLOWED_ORIGINS")
    if parts.scheme != "https" and parts.hostname != "127.0.0.1":
        raise ValueError("Remote endpoints require HTTPS")
    return url.rstrip("/")


async def request_json(url: str, *, payload=None, api_key="", max_bytes=2_000_000):
    validate_url(url)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=90, follow_redirects=False, trust_env=False) as client:
            async with client.stream("POST" if payload is not None else "GET", url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    raise ValueError(f"Provider returned HTTP {response.status_code}; check access, quota and endpoint")
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > max_bytes:
                        raise ValueError("Provider response exceeds size limit")
    except httpx.HTTPError:
        raise ValueError("Provider connection failed") from None
    return json.loads(content)


async def remote_generate(profile: dict, messages: list[dict], max_tokens: int) -> str:
    body = {"model": profile["model"], "messages": messages, "max_tokens": min(max_tokens, 8192), "stream": False}
    data = await request_json(profile["url"] + "/chat/completions", payload=body, api_key=profile.get("api_key", ""))
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
        if choice.get("finish_reason") == "length":
            raise ValueError("Provider output was truncated; no tool proposal accepted")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Provider did not return text")
        return content
    except (KeyError, IndexError, TypeError):
        raise ValueError("Unsupported provider response") from None
