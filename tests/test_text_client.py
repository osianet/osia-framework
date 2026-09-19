"""Tests for src/intelligence/text_client.py — the Google-free text abstraction."""

from __future__ import annotations

import httpx
import pytest

from src.intelligence.text_client import (
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_VENICE_MODEL,
    TextClient,
    TextError,
)

_ENV_KEYS = [
    "OSIA_TEXT_PROVIDERS",
    "OSIA_TEXT_ALLOW_GEMINI",
    "OSIA_TEXT_VENICE_MODEL",
    "OSIA_TEXT_OPENROUTER_MODEL",
    "VENICE_API_KEY",
    "OPENROUTER_API_KEY",
    "GEMINI_API_KEY",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    yield


def test_default_cascade(monkeypatch):
    monkeypatch.setenv("VENICE_API_KEY", "vk")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ok")
    tc = TextClient()
    names = [(p.name, p.model) for p in tc._providers]
    assert names == [("venice", DEFAULT_VENICE_MODEL), ("openrouter", DEFAULT_OPENROUTER_MODEL)]
    assert tc.available is True


def test_no_keys_unavailable(monkeypatch):
    tc = TextClient()
    assert tc.available is False


def test_gemini_opt_in_gate(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gk")
    monkeypatch.setenv("OSIA_TEXT_PROVIDERS", "gemini")
    assert TextClient()._providers == []  # allow flag off
    monkeypatch.setenv("OSIA_TEXT_ALLOW_GEMINI", "1")
    assert [p.name for p in TextClient()._providers] == ["gemini"]


async def test_generate_first_success(monkeypatch):
    monkeypatch.setenv("VENICE_API_KEY", "vk")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ok")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"choices": [{"message": {"content": "HELLO"}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tc = TextClient(http_client=client)
    out = await tc.generate("hi")
    assert out == "HELLO"
    assert len(seen) == 1 and "venice.ai" in seen[0]
    await client.aclose()


async def test_generate_cascades(monkeypatch):
    monkeypatch.setenv("VENICE_API_KEY", "vk")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ok")

    def handler(request: httpx.Request) -> httpx.Response:
        if "venice.ai" in str(request.url):
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "OR-OK"}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tc = TextClient(http_client=client)
    assert await tc.generate("hi") == "OR-OK"
    await client.aclose()


async def test_generate_all_fail(monkeypatch):
    monkeypatch.setenv("VENICE_API_KEY", "vk")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tc = TextClient(http_client=client)
    with pytest.raises(TextError):
        await tc.generate("hi")
    await client.aclose()


async def test_no_providers_raises(monkeypatch):
    tc = TextClient()
    with pytest.raises(TextError, match="No text providers"):
        await tc.generate("hi")


async def test_system_prompt_included(monkeypatch):
    monkeypatch.setenv("VENICE_API_KEY", "vk")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tc = TextClient(http_client=client)
    await tc.generate("do it", system="you are X", temperature=0.7, max_tokens=99)
    body = captured["body"]
    assert body["messages"][0] == {"role": "system", "content": "you are X"}
    assert body["messages"][1] == {"role": "user", "content": "do it"}
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 99
    await client.aclose()
