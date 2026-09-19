"""Tests for src/intelligence/vision_client.py — the Google-free vision abstraction.

Covers:
* provider cascade assembly from env (default, custom order, gemini opt-in gate)
* image / frame data-URL encoding
* the dispatch cascade: fallthrough on transient error, stop on first success,
  correct payload shape — all with httpx mocked (no network)
* real ffmpeg frame sampling from a tiny generated test clip (skipped if no ffmpeg)
"""

from __future__ import annotations

import base64
import shutil
import subprocess

import httpx
import pytest

from src.intelligence.vision_client import (
    DEFAULT_VENICE_MODEL,
    VisionClient,
    VisionError,
    _is_transient,
)

# ---------------------------------------------------------------------------
# env fixture — isolate every test from the real environment
# ---------------------------------------------------------------------------

_VISION_ENV_KEYS = [
    "OSIA_VISION_PROVIDERS",
    "OSIA_VISION_ALLOW_GEMINI",
    "OSIA_VISION_VENICE_MODEL",
    "OSIA_VISION_OPENROUTER_MODELS",
    "OSIA_VISION_MAX_FRAMES",
    "OSIA_VISION_FRAME_FPS",
    "OSIA_VISION_GEMINI_MODEL",
    "VENICE_API_KEY",
    "OPENROUTER_API_KEY",
    "GEMINI_API_KEY",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _VISION_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    yield


# ---------------------------------------------------------------------------
# Provider assembly
# ---------------------------------------------------------------------------


def test_default_cascade_is_venice_then_openrouter(monkeypatch):
    monkeypatch.setenv("VENICE_API_KEY", "vk")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ok")
    vc = VisionClient()
    names = [p.name for p in vc._providers]
    assert names == ["venice", "openrouter"]
    assert vc._providers[0].models[0] == DEFAULT_VENICE_MODEL
    assert vc.available is True


def test_no_keys_means_no_providers(monkeypatch):
    vc = VisionClient()
    assert vc._providers == []
    assert vc.available is False


def test_gemini_is_opt_in_only(monkeypatch):
    """gemini must never appear unless explicitly listed AND allowed."""
    monkeypatch.setenv("GEMINI_API_KEY", "gk")
    monkeypatch.setenv("OSIA_VISION_PROVIDERS", "gemini")
    # allow flag off → gemini excluded
    vc = VisionClient()
    assert vc._providers == []
    # allow flag on → gemini included
    monkeypatch.setenv("OSIA_VISION_ALLOW_GEMINI", "1")
    vc2 = VisionClient()
    assert [p.name for p in vc2._providers] == ["gemini"]


def test_gemini_not_in_default_cascade_even_when_allowed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gk")
    monkeypatch.setenv("VENICE_API_KEY", "vk")
    monkeypatch.setenv("OSIA_VISION_ALLOW_GEMINI", "1")
    vc = VisionClient()  # default OSIA_VISION_PROVIDERS = "venice,openrouter"
    assert "gemini" not in [p.name for p in vc._providers]


def test_custom_order_and_models(monkeypatch):
    monkeypatch.setenv("VENICE_API_KEY", "vk")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ok")
    monkeypatch.setenv("OSIA_VISION_PROVIDERS", "openrouter,venice")
    monkeypatch.setenv("OSIA_VISION_OPENROUTER_MODELS", "a/model-1, b/model-2")
    vc = VisionClient()
    assert [p.name for p in vc._providers] == ["openrouter", "venice"]
    assert vc._providers[0].models == ("a/model-1", "b/model-2")


def test_openrouter_carries_referer_headers(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "ok")
    monkeypatch.setenv("OSIA_VISION_PROVIDERS", "openrouter")
    vc = VisionClient()
    headers = vc._providers[0].headers
    assert headers["Authorization"] == "Bearer ok"
    assert headers["HTTP-Referer"] == "https://osia.dev"


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_image_data_url_roundtrip(tmp_path):
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    url = VisionClient._image_data_url(img)
    assert url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(url.split(",", 1)[1])
    assert decoded == b"\x89PNG\r\n\x1a\nFAKE"


def test_image_data_url_jpg_mime(tmp_path):
    img = tmp_path / "shot.jpg"
    img.write_bytes(b"\xff\xd8\xff")
    url = VisionClient._image_data_url(img)
    assert url.startswith("data:image/jpeg;base64,")


def test_missing_image_raises(tmp_path):
    with pytest.raises(VisionError):
        VisionClient._image_data_url(tmp_path / "nope.png")


# ---------------------------------------------------------------------------
# Transient classifier
# ---------------------------------------------------------------------------


def test_is_transient_status_error():
    resp = httpx.Response(503, request=httpx.Request("POST", "http://x"))
    exc = httpx.HTTPStatusError("boom", request=resp.request, response=resp)
    assert _is_transient(exc) is True
    resp2 = httpx.Response(401, request=httpx.Request("POST", "http://x"))
    exc2 = httpx.HTTPStatusError("nope", request=resp2.request, response=resp2)
    assert _is_transient(exc2) is False


def test_is_transient_string_match():
    assert _is_transient(RuntimeError("Connection reset by peer")) is True
    assert _is_transient(ValueError("bad prompt")) is False


# ---------------------------------------------------------------------------
# Dispatch cascade (httpx mocked)
# ---------------------------------------------------------------------------


def _mock_transport(handler):
    return httpx.MockTransport(handler)


async def test_dispatch_first_success(monkeypatch):
    monkeypatch.setenv("VENICE_API_KEY", "vk")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ok")

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"choices": [{"message": {"content": "VENICE SAW IT"}}]})

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    vc = VisionClient(http_client=client)
    out = await vc.analyse_image_bytes_for_test(b"img")
    assert out == "VENICE SAW IT"
    # only the first provider was called
    assert len(calls) == 1
    assert "venice.ai" in calls[0]
    await client.aclose()


async def test_dispatch_falls_through_transient(monkeypatch):
    monkeypatch.setenv("VENICE_API_KEY", "vk")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ok")
    monkeypatch.setenv("OSIA_VISION_OPENROUTER_MODELS", "or/model-a")

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if "venice.ai" in str(request.url):
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "OPENROUTER SAVED"}}]})

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    vc = VisionClient(http_client=client)
    out = await vc.analyse_image_bytes_for_test(b"img")
    assert out == "OPENROUTER SAVED"
    assert len(seen) == 2  # venice failed, openrouter succeeded
    assert "venice.ai" in seen[0]
    assert "openrouter.ai" in seen[1]
    await client.aclose()


async def test_dispatch_all_fail_raises(monkeypatch):
    monkeypatch.setenv("VENICE_API_KEY", "vk")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    vc = VisionClient(http_client=client)
    with pytest.raises(VisionError):
        await vc.analyse_image_bytes_for_test(b"img")
    await client.aclose()


async def test_no_providers_raises(monkeypatch):
    client = httpx.AsyncClient(transport=_mock_transport(lambda r: httpx.Response(200)))
    vc = VisionClient(http_client=client)
    with pytest.raises(VisionError, match="No vision providers"):
        await vc.analyse_image_bytes_for_test(b"img")
    await client.aclose()


async def test_payload_shape_is_openai_compatible(monkeypatch):
    monkeypatch.setenv("VENICE_API_KEY", "vk")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    vc = VisionClient(http_client=client)
    await vc.analyse_image_bytes_for_test(b"img", prompt="What is this?")
    body = captured["body"]
    assert captured["auth"] == "Bearer vk"
    assert body["model"] == DEFAULT_VENICE_MODEL
    msg = body["messages"][0]
    assert msg["role"] == "user"
    # media part first, text part last
    assert msg["content"][0]["type"] == "image_url"
    assert msg["content"][-1] == {"type": "text", "text": "What is this?"}
    await client.aclose()


# ---------------------------------------------------------------------------
# Real ffmpeg frame sampling
# ---------------------------------------------------------------------------

_HAS_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")
def test_frame_sampling_real(tmp_path, monkeypatch):
    monkeypatch.setenv("VENICE_API_KEY", "vk")
    monkeypatch.setenv("OSIA_VISION_MAX_FRAMES", "4")
    monkeypatch.setenv("OSIA_VISION_FRAME_FPS", "1")
    # generate a 3-second test clip
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=320x240:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        capture_output=True,
        check=True,
    )
    vc = VisionClient()
    frames = vc._sample_frames(clip)
    assert 1 <= len(frames) <= 4
    # each frame is a JPEG (starts with the JPEG SOI marker)
    for f in frames:
        assert f[:2] == b"\xff\xd8"


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")
def test_frame_sampling_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("VENICE_API_KEY", "vk")
    vc = VisionClient()
    with pytest.raises(VisionError):
        vc._sample_frames(tmp_path / "does_not_exist.mp4")
