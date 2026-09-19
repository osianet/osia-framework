"""
OSIA Text Client — provider-agnostic plain text generation.

A minimal, Google-free companion to :mod:`src.intelligence.vision_client` for the
handful of call sites that just need "prompt in -> text out" and were previously
pinned to the Google genai SDK. Uses OpenAI-compatible ``/v1/chat/completions``,
so one code path spans Venice, OpenRouter, DashScope, etc.

Cascade defaults to Venice -> OpenRouter (entirely Google-free). Gemini is only an
opt-in last resort, enabled with ``OSIA_TEXT_ALLOW_GEMINI=1`` AND listed in
``OSIA_TEXT_PROVIDERS``. It is never tried first and never required.

Environment variables
----------------------
  OSIA_TEXT_PROVIDERS        Comma-separated cascade (default: "venice,openrouter").
  OSIA_TEXT_ALLOW_GEMINI     "1"/"true" to permit the gemini backend at all.
  OSIA_TEXT_VENICE_MODEL     Venice model id (default: venice-uncensored).
  OSIA_TEXT_OPENROUTER_MODEL OpenRouter model id (default: qwen/qwen3.8-27b).
  OSIA_TEXT_GEMINI_MODEL     Gemini model id via its OpenAI-compat endpoint.
  VENICE_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY  provider credentials.

Usage:
    tc = TextClient()
    text = await tc.generate("Summarise this article: ...", temperature=0.3)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger("osia.text")

DEFAULT_VENICE_MODEL = "venice-uncensored"
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3.8-27b"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.3
REQUEST_TIMEOUT = 120.0

_VENICE_BASE = "https://api.venice.ai/api"
_OPENROUTER_BASE = "https://openrouter.ai/api"
_GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_OPENROUTER_HEADERS = {"HTTP-Referer": "https://osia.dev", "X-Title": "OSIA Intelligence Framework"}


class TextError(RuntimeError):
    """Raised when every configured text provider fails."""


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class _Provider:
    name: str
    base_url: str
    api_key: str
    model: str
    extra_headers: tuple[tuple[str, str], ...] = ()

    @property
    def headers(self) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        h.update(dict(self.extra_headers))
        return h


class TextClient:
    """Provider-agnostic plain text generation via OpenAI-compatible APIs."""

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._http = http_client
        self._owns_http = http_client is None
        self._providers = self._build_providers()

    def _build_providers(self) -> list[_Provider]:
        order = [
            p.strip().lower() for p in os.getenv("OSIA_TEXT_PROVIDERS", "venice,openrouter").split(",") if p.strip()
        ]
        allow_gemini = _env_flag("OSIA_TEXT_ALLOW_GEMINI", False)
        providers: list[_Provider] = []
        for name in order:
            if name == "venice":
                key = os.getenv("VENICE_API_KEY", "")
                if key:
                    providers.append(
                        _Provider(
                            "venice", _VENICE_BASE, key, os.getenv("OSIA_TEXT_VENICE_MODEL", DEFAULT_VENICE_MODEL)
                        )
                    )
            elif name == "openrouter":
                key = os.getenv("OPENROUTER_API_KEY", "")
                if key:
                    providers.append(
                        _Provider(
                            "openrouter",
                            _OPENROUTER_BASE,
                            key,
                            os.getenv("OSIA_TEXT_OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
                            tuple(_OPENROUTER_HEADERS.items()),
                        )
                    )
            elif name == "gemini":
                if allow_gemini and os.getenv("GEMINI_API_KEY"):
                    providers.append(
                        _Provider(
                            "gemini",
                            _GEMINI_OPENAI_BASE,
                            os.getenv("GEMINI_API_KEY", ""),
                            os.getenv("OSIA_TEXT_GEMINI_MODEL", "gemini-2.5-flash"),
                        )
                    )
        return providers

    @property
    def available(self) -> bool:
        return bool(self._providers)

    def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
            self._owns_http = True
        return self._http

    async def aclose(self) -> None:
        if self._owns_http and self._http and not self._http.is_closed:
            await self._http.aclose()

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Generate text from a prompt, cascading across providers. Returns the text."""
        if not self._providers:
            raise TextError("No text providers configured. Set VENICE_API_KEY and/or OPENROUTER_API_KEY.")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_exc: Exception | None = None
        for provider in self._providers:
            try:
                payload = {
                    "model": provider.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                resp = await self._client().post(
                    f"{provider.base_url}/v1/chat/completions",
                    headers=provider.headers,
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 — cascade on any failure
                last_exc = exc
                logger.warning("Text: %s/%s failed (%s) — trying next.", provider.name, provider.model, str(exc)[:160])
        raise TextError(f"All text providers exhausted; last error: {last_exc}") from last_exc
