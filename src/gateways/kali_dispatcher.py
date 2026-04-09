"""
KaliDispatcher — thin async HTTP client for the local Kali Linux API.

The Kali API runs in a Docker container on 127.0.0.1:8100 and exposes
offensive/recon tooling (nmap, whois, dig, sslscan, etc.) as REST endpoints.
All tool endpoints require a Bearer token from KALI_API_KEY in .env.

Usage:
    dispatcher = KaliDispatcher()
    result = await dispatcher.call_tool("nmap", {"target": "10.0.0.1", "top_ports": 100})
    await dispatcher.close()
"""

import logging
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("osia.kali_dispatcher")

KALI_BASE_URL = "http://127.0.0.1:8100"

# Per-tool read timeouts in seconds — generous to cover worst-case runs.
# Amass can run up to 30 min; nikto up to 600s; everything else is much shorter.
_TOOL_TIMEOUTS: dict[str, float] = {
    "nmap": 660.0,
    "whois": 30.0,
    "dig": 30.0,
    "ping": 30.0,
    "curl": 70.0,
    "traceroute": 200.0,
    "nikto": 660.0,
    "harvester": 200.0,
    "whatweb": 60.0,
    "sslscan": 45.0,
    "amass": 1830.0,
}
_DEFAULT_TIMEOUT = 120.0


class KaliDispatcher:
    """Async client for the local Kali Linux API container."""

    def __init__(self) -> None:
        self._api_key = os.getenv("KALI_API_KEY", "")
        if not self._api_key:
            logger.warning("KALI_API_KEY is not set — Kali tool calls will be rejected by the API.")
        # Separate client per call so timeouts are per-request, not shared.
        # Using a factory instead of a single shared client avoids cross-tool timeout pollution.
        self._connect_timeout = 10.0

    def _make_client(self, tool: str) -> httpx.AsyncClient:
        timeout = _TOOL_TIMEOUTS.get(tool, _DEFAULT_TIMEOUT)
        return httpx.AsyncClient(
            timeout=httpx.Timeout(connect=self._connect_timeout, read=timeout, write=30.0, pool=5.0),
        )

    async def call_tool(self, tool: str, payload: dict) -> dict:
        """
        POST to /tools/<tool> and return the response JSON.

        On HTTP error or network failure the exception is re-raised so the
        orchestrator's research loop can log it and continue.
        """
        url = f"{KALI_BASE_URL}/tools/{tool}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        logger.info("Kali tool '%s' → target=%s", tool, payload.get("target") or payload.get("url", "?"))
        async with self._make_client(tool) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def health_check(self) -> bool:
        """Return True if the Kali API container is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{KALI_BASE_URL}/health")
                return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """No persistent client to close — kept for API symmetry with MCPDispatcher."""
