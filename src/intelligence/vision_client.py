"""
OSIA Vision Client — provider-agnostic image & video comprehension.

This module replaces OSIA's direct dependence on Google Gemini for multimodal
(image / video) understanding. It uses the **frame-sampling → OpenAI-compatible
`image_url`** technique, which works against every OpenAI-compatible vision
backend (Venice, OpenRouter, DashScope, Anthropic-compat, …) with a single code
path, so no single provider — Google included — is load-bearing any more.

Design
------
* **Images** are base64-encoded and sent as one ``image_url`` content part.
* **Videos** are transcoded/sampled with ffmpeg into N JPEG frames, each sent as
  an ``image_url`` part alongside the prompt. Optionally an audio transcript can
  be extracted and prepended (callers pass ``transcript=`` when they have one).
* Providers are tried in a **cascade**. The default order is
  Venice → OpenRouter, i.e. entirely Google-free. Google Gemini is available only
  as an explicit, opt-in *last-resort* backend, enabled with
  ``OSIA_VISION_ALLOW_GEMINI=1``. It is never tried first and never required.

Environment variables
----------------------
  OSIA_VISION_PROVIDERS       Comma-separated cascade (default: "venice,openrouter").
                              Recognised: venice, openrouter, gemini.
  OSIA_VISION_ALLOW_GEMINI    "1"/"true" to permit the gemini backend at all
                              (default: off). Even when allowed it is only used
                              if listed in OSIA_VISION_PROVIDERS.
  OSIA_VISION_VENICE_MODEL    Venice vision model id (default: qwen3-vl-235b-a22b).
  OSIA_VISION_OPENROUTER_MODELS
                              Comma-separated OpenRouter vision model ids, tried
                              in order (default: a spread across Anthropic / Qwen /
                              OpenAI so one vendor outage cannot take vision down).
  OSIA_VISION_MAX_FRAMES      Max frames sampled from a video (default: 16).
  OSIA_VISION_FRAME_FPS       Frame sampling rate, frames/sec (default: 1.0).
  OSIA_VISION_FRAME_LONG_EDGE Downscale long edge of each frame, px (default: 768).
  OSIA_VISION_VIDEO_MAX_SECS  Hard cap on video seconds sampled (default: 180).
  VENICE_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY
                              Provider credentials (only the ones in the cascade
                              are needed).

The public surface is intentionally small:

    vc = VisionClient()
    text = await vc.analyse_image(path, prompt)
    text = await vc.analyse_video(path, prompt, transcript=optional_str)

Both return the model's text response (str). On total failure they raise
``VisionError`` with the last provider exception chained.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import subprocess  # noqa: S404 — ffmpeg invocation is intentional
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger("osia.vision")

# ---------------------------------------------------------------------------
# Defaults / config
# ---------------------------------------------------------------------------

DEFAULT_VENICE_MODEL = "qwen3-vl-235b-a22b"
DEFAULT_OPENROUTER_MODELS = [
    "anthropic/claude-sonnet-5",  # strong OCR + reasoning
    "qwen/qwen3.8-27b",  # open-weight VLM, cheap, solid vision
    "openai/gpt-5.6-luna",  # different vendor entirely — outage isolation
]
DEFAULT_MAX_FRAMES = 16
DEFAULT_FRAME_FPS = 1.0
DEFAULT_FRAME_LONG_EDGE = 768
DEFAULT_VIDEO_MAX_SECS = 180
REQUEST_TIMEOUT = 180.0
DEFAULT_MAX_TOKENS = 1536

_VENICE_BASE = "https://api.venice.ai/api"
_OPENROUTER_BASE = "https://openrouter.ai/api"
_OPENROUTER_HEADERS = {"HTTP-Referer": "https://osia.dev", "X-Title": "OSIA Intelligence Framework"}


class VisionError(RuntimeError):
    """Raised when every configured vision provider fails."""


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _is_transient(exc: Exception) -> bool:
    """Return True for errors that warrant trying the next provider."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (408, 409, 429, 500, 502, 503, 504)
    msg = str(exc).upper()
    return any(
        tok in msg
        for tok in (
            "429",
            "503",
            "502",
            "504",
            "UNAVAILABLE",
            "RESOURCE_EXHAUSTED",
            "OVERLOADED",
            "SSL",
            "HANDSHAKE",
            "TIMED OUT",
            "TIMEOUT",
            "CONNECTION",
            "RESET",
            "EOF",
        )
    )


# ---------------------------------------------------------------------------
# Provider descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Provider:
    name: str
    base_url: str
    api_key: str
    models: tuple[str, ...]
    extra_headers: tuple[tuple[str, str], ...] = ()

    @property
    def headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        h.update(dict(self.extra_headers))
        return h


# ---------------------------------------------------------------------------
# VisionClient
# ---------------------------------------------------------------------------


class VisionClient:
    """Provider-agnostic image/video comprehension via OpenAI-compatible APIs."""

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._http = http_client
        self._owns_http = http_client is None
        self.max_tokens = max_tokens

        self.max_frames = int(os.getenv("OSIA_VISION_MAX_FRAMES", str(DEFAULT_MAX_FRAMES)))
        self.frame_fps = float(os.getenv("OSIA_VISION_FRAME_FPS", str(DEFAULT_FRAME_FPS)))
        self.frame_long_edge = int(os.getenv("OSIA_VISION_FRAME_LONG_EDGE", str(DEFAULT_FRAME_LONG_EDGE)))
        self.video_max_secs = int(os.getenv("OSIA_VISION_VIDEO_MAX_SECS", str(DEFAULT_VIDEO_MAX_SECS)))

        self._providers = self._build_providers()
        if not self._providers:
            logger.warning(
                "VisionClient: no usable providers configured. Set VENICE_API_KEY or "
                "OPENROUTER_API_KEY (and OSIA_VISION_PROVIDERS if customising the cascade)."
            )
        else:
            logger.info(
                "VisionClient cascade: %s",
                " → ".join(f"{p.name}({p.models[0]})" for p in self._providers),
            )

    # ------------------------------------------------------------------
    # Provider assembly
    # ------------------------------------------------------------------

    def _build_providers(self) -> list[_Provider]:
        order = [
            p.strip().lower() for p in os.getenv("OSIA_VISION_PROVIDERS", "venice,openrouter").split(",") if p.strip()
        ]
        allow_gemini = _env_flag("OSIA_VISION_ALLOW_GEMINI", False)
        providers: list[_Provider] = []

        for name in order:
            if name == "venice":
                key = os.getenv("VENICE_API_KEY", "")
                if not key:
                    logger.debug("VisionClient: skipping venice (no VENICE_API_KEY).")
                    continue
                model = os.getenv("OSIA_VISION_VENICE_MODEL", DEFAULT_VENICE_MODEL)
                providers.append(_Provider("venice", _VENICE_BASE, key, (model,)))
            elif name == "openrouter":
                key = os.getenv("OPENROUTER_API_KEY", "")
                if not key:
                    logger.debug("VisionClient: skipping openrouter (no OPENROUTER_API_KEY).")
                    continue
                raw = os.getenv("OSIA_VISION_OPENROUTER_MODELS", "")
                models = tuple(m.strip() for m in raw.split(",") if m.strip()) or tuple(DEFAULT_OPENROUTER_MODELS)
                providers.append(
                    _Provider(
                        "openrouter",
                        _OPENROUTER_BASE,
                        key,
                        models,
                        tuple(_OPENROUTER_HEADERS.items()),
                    )
                )
            elif name == "gemini":
                if not allow_gemini:
                    logger.debug("VisionClient: gemini requested but OSIA_VISION_ALLOW_GEMINI is off — skipping.")
                    continue
                key = os.getenv("GEMINI_API_KEY", "")
                if not key:
                    logger.debug("VisionClient: skipping gemini (no GEMINI_API_KEY).")
                    continue
                # Gemini exposes an OpenAI-compatible endpoint — no genai SDK needed.
                model = os.getenv("OSIA_VISION_GEMINI_MODEL", "gemini-2.5-flash")
                providers.append(
                    _Provider(
                        "gemini",
                        "https://generativelanguage.googleapis.com/v1beta/openai",
                        key,
                        (model,),
                    )
                )
            else:
                logger.warning("VisionClient: unknown provider '%s' in OSIA_VISION_PROVIDERS — ignoring.", name)

        return providers

    # ------------------------------------------------------------------
    # HTTP client
    # ------------------------------------------------------------------

    def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
            self._owns_http = True
        return self._http

    async def aclose(self) -> None:
        if self._owns_http and self._http and not self._http.is_closed:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if at least one provider is configured."""
        return bool(self._providers)

    async def analyse_image(self, image_path: str | Path, prompt: str) -> str:
        """Describe/analyse a single image. Returns the model's text response."""
        data_url = self._image_data_url(Path(image_path))
        parts = [{"type": "image_url", "image_url": {"url": data_url}}]
        return await self._dispatch(prompt, parts)

    async def analyse_image_bytes(
        self, data: bytes, prompt: str = "Describe this image.", mime_subtype: str = "png"
    ) -> str:
        """Describe/analyse an in-memory image (no filesystem path required)."""
        parts = [{"type": "image_url", "image_url": {"url": self._bytes_data_url(data, mime_subtype)}}]
        return await self._dispatch(prompt, parts)

    # Test-facing alias kept stable for the unit suite.
    analyse_image_bytes_for_test = analyse_image_bytes

    async def analyse_video(
        self,
        video_path: str | Path,
        prompt: str,
        transcript: str | None = None,
    ) -> str:
        """Sample frames from a video and analyse them. Returns text response.

        ``transcript`` — optional pre-extracted audio transcript. Since none of
        the frame-based vision backends hear audio, callers that already have a
        transcript (yt-dlp captions, Whisper, ADB) should pass it so speech is
        not lost. It is prepended to the prompt as context.
        """
        frames = await asyncio.to_thread(self._sample_frames, Path(video_path))
        if not frames:
            raise VisionError(f"Could not sample any frames from video: {video_path}")

        parts = [{"type": "image_url", "image_url": {"url": self._bytes_data_url(f, "jpeg")}} for f in frames]

        full_prompt = prompt
        if transcript and transcript.strip():
            full_prompt = (
                f"AUDIO TRANSCRIPT (extracted separately — the frames below are silent):\n"
                f"{transcript.strip()}\n\n{prompt}"
            )
        else:
            full_prompt = (
                f"The following {len(frames)} images are frames sampled in chronological order "
                f"from a video (the frames are silent — no audio is available).\n\n{prompt}"
            )
        return await self._dispatch(full_prompt, parts)

    # ------------------------------------------------------------------
    # Frame sampling (ffmpeg)
    # ------------------------------------------------------------------

    def _sample_frames(self, video_path: Path) -> list[bytes]:
        """Extract up to ``max_frames`` JPEG frames from a video via ffmpeg.

        Runs synchronously (call via asyncio.to_thread). Returns a list of JPEG
        byte blobs in chronological order. Frames are downscaled so the long edge
        is at most ``frame_long_edge`` to keep token cost bounded.
        """
        if not video_path.exists():
            raise VisionError(f"Video file not found: {video_path}")

        with tempfile.TemporaryDirectory(prefix="osia_vision_", dir=os.getenv("KIROCREW_SCRATCH") or None) as tmp:
            out_pattern = str(Path(tmp) / "frame_%04d.jpg")
            # fps=N/1 sampling + scale long edge, capped at video_max_secs.
            vf = (
                f"fps={self.frame_fps},"
                f"scale='if(gt(iw,ih),{self.frame_long_edge},-2)':'if(gt(iw,ih),-2,{self.frame_long_edge})'"
            )
            cmd = [
                "ffmpeg",
                "-y",
                "-t",
                str(self.video_max_secs),
                "-i",
                str(video_path),
                "-vf",
                vf,
                "-vsync",
                "vfr",
                "-frames:v",
                str(self.max_frames),
                "-q:v",
                "4",
                out_pattern,
            ]
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                timeout=self.video_max_secs + 120,
                check=False,
            )
            if proc.returncode != 0:
                logger.warning(
                    "ffmpeg frame sampling failed (rc=%d): %s",
                    proc.returncode,
                    proc.stderr.decode(errors="replace")[:400],
                )
                return []

            frame_files = sorted(Path(tmp).glob("frame_*.jpg"))
            if len(frame_files) > self.max_frames:
                # Evenly subsample down to max_frames if ffmpeg over-produced.
                step = len(frame_files) / self.max_frames
                frame_files = [frame_files[int(i * step)] for i in range(self.max_frames)]
            return [f.read_bytes() for f in frame_files]

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _image_data_url(path: Path) -> str:
        if not path.exists():
            raise VisionError(f"Image file not found: {path}")
        suffix = path.suffix.lower().lstrip(".") or "png"
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}.get(suffix, "png")
        return VisionClient._bytes_data_url(path.read_bytes(), mime)

    @staticmethod
    def _bytes_data_url(data: bytes, mime_subtype: str) -> str:
        b64 = base64.b64encode(data).decode()
        return f"data:image/{mime_subtype};base64,{b64}"

    # ------------------------------------------------------------------
    # Dispatch cascade
    # ------------------------------------------------------------------

    async def _dispatch(self, prompt: str, media_parts: list[dict]) -> str:
        """Try each provider/model in the cascade until one succeeds."""
        if not self._providers:
            raise VisionError("No vision providers configured. Set VENICE_API_KEY and/or OPENROUTER_API_KEY.")

        content = [*media_parts, {"type": "text", "text": prompt}]
        last_exc: Exception | None = None

        for provider in self._providers:
            for model in provider.models:
                try:
                    text = await self._call(provider, model, content)
                    if provider.name != self._providers[0].name or model != self._providers[0].models[0]:
                        logger.info("Vision: succeeded via %s/%s (fallback).", provider.name, model)
                    return text
                except Exception as exc:  # noqa: BLE001 — cascade on any failure
                    last_exc = exc
                    if _is_transient(exc):
                        logger.warning(
                            "Vision: %s/%s transient failure (%s) — trying next.",
                            provider.name,
                            model,
                            str(exc)[:160],
                        )
                    else:
                        logger.warning(
                            "Vision: %s/%s failed (%s) — trying next.",
                            provider.name,
                            model,
                            str(exc)[:160],
                        )
        raise VisionError(f"All vision providers exhausted; last error: {last_exc}") from last_exc

    async def _call(self, provider: _Provider, model: str, content: list[dict]) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
        }
        resp = await self._client().post(
            f"{provider.base_url}/v1/chat/completions",
            headers=provider.headers,
            json=payload,
        )
        resp.raise_for_status()
        body = resp.json()
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionError(f"{provider.name}/{model}: unexpected response shape: {str(body)[:200]}") from exc
