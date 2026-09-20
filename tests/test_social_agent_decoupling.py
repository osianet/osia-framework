"""Regression guard: SocialMediaAgent must not require the Google genai SDK.

Before the de-Google work, SocialMediaAgent took a *required* ``gemini_client``
and hard-imported ``from google import genai`` at module top. These tests pin the
new contract: it constructs with no Gemini client, and its vision path uses the
injected VisionClient rather than Gemini.
"""

from __future__ import annotations

import httpx
import pytest

from src.agents.social_media_agent import SocialMediaAgent
from src.intelligence.vision_client import VisionClient


class _FakeADB:
    """Minimal ADB stand-in — SocialMediaAgent only stores it at construction."""

    device_id = "fake"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("VENICE_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "OSIA_VISION_PROVIDERS"):
        monkeypatch.delenv(k, raising=False)
    yield


def test_constructs_without_gemini_client(monkeypatch, tmp_path):
    monkeypatch.setenv("VENICE_API_KEY", "vk")
    agent = SocialMediaAgent(adb=_FakeADB(), base_dir=tmp_path)
    assert agent.gemini is None
    assert agent.vision.available is True


async def test_vision_path_uses_vision_client(monkeypatch, tmp_path):
    """_generate_with_fallback must reach VisionClient, not Gemini, when no genai client."""
    monkeypatch.setenv("VENICE_API_KEY", "vk")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "SCREEN DESCRIBED"}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    vc = VisionClient(http_client=client)
    agent = SocialMediaAgent(adb=_FakeADB(), base_dir=tmp_path, vision_client=vc)

    shot = tmp_path / "screen.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    out = await agent._generate_with_fallback(str(shot), "What is on screen?")
    assert out == "SCREEN DESCRIBED"
    await client.aclose()
