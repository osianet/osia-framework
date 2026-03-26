"""
HuggingFace Inference Endpoints manager for OSIA.

Handles waking up scale-to-zero endpoints before dispatching queries to
AnythingLLM desks that use HF-hosted uncensored models.

The orchestrator calls `ensure_ready(desk_slug)` after routing — if the desk
uses an HF endpoint, this wakes it up and blocks until it's serving.
"""

import asyncio
import logging
import os

import httpx

logger = logging.getLogger("osia.hf_endpoints")

# Seconds to wait after "running" status for the model to actually serve requests
READINESS_PROBE_TIMEOUT = 60
READINESS_PROBE_INTERVAL = 5

# Map desk slugs → HF endpoint names. Only desks backed by HF endpoints
# need entries here. All other desks are ignored (no-op).
DESK_ENDPOINT_MAP: dict[str, str] = {
    "human-intelligence-and-profiling-desk": "osia-dolphin-r1-24b",
    "cyber-intelligence-and-warfare-desk": "osia-hermes-70b",
    "cultural-and-theological-intelligence-desk": "osia-dolphin-r1-24b",
}

# How long to wait for an endpoint to become ready before giving up
WAKE_TIMEOUT_SECONDS = int(os.getenv("HF_WAKE_TIMEOUT", "600"))
POLL_INTERVAL_SECONDS = 10


class HFEndpointManager:
    """Manages HuggingFace Inference Endpoint lifecycle for OSIA desks."""

    def __init__(self):
        self.token = os.getenv("HF_TOKEN")
        self.namespace = os.getenv("HF_NAMESPACE")
        self.enabled = bool(self.token and self.namespace)

        if not self.enabled:
            logger.info("HF Endpoints disabled (HF_TOKEN or HF_NAMESPACE not set)")

    async def ensure_ready(self, desk_slug: str) -> bool:
        """
        If this desk uses an HF endpoint, wake it up and wait until it's serving.
        Returns True if the endpoint is ready (or desk doesn't need one).
        Returns False if wake-up failed or timed out.
        """
        if not self.enabled:
            return True

        endpoint_name = DESK_ENDPOINT_MAP.get(desk_slug)
        if not endpoint_name:
            return True  # desk doesn't use an HF endpoint

        logger.info("Desk '%s' uses HF endpoint '%s' — checking status...", desk_slug, endpoint_name)

        try:
            url = await asyncio.to_thread(self._wake_and_wait, endpoint_name)
            if not url:
                return False
            # Endpoint reports "running" but model may not be warm yet — probe it
            return await self._probe_until_ready(endpoint_name, url)
        except Exception as e:
            logger.error("Failed to wake HF endpoint '%s': %s", endpoint_name, e)
            return False

    async def _probe_until_ready(self, endpoint_name: str, url: str) -> bool:
        """Hit the inference URL until it actually responds, or time out."""
        logger.info("Probing endpoint '%s' at %s for model readiness...", endpoint_name, url)
        deadline = asyncio.get_event_loop().time() + READINESS_PROBE_TIMEOUT
        async with httpx.AsyncClient(timeout=10.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.post(
                        f"{url}/v1/models",
                        headers={"Authorization": f"Bearer {self.token}"},
                    )
                    if resp.status_code < 500:
                        logger.info("Endpoint '%s' model is warm and serving.", endpoint_name)
                        return True
                except (httpx.ConnectError, httpx.ReadTimeout):
                    pass
                logger.debug("Endpoint '%s' not yet serving, retrying in %ds...", endpoint_name, READINESS_PROBE_INTERVAL)
                await asyncio.sleep(READINESS_PROBE_INTERVAL)
        logger.error("Endpoint '%s' reported running but model never became ready within %ds", endpoint_name, READINESS_PROBE_TIMEOUT)
        return False

    def _wake_and_wait(self, endpoint_name: str) -> str | None:
        """Synchronous blocking call (run via to_thread) that wakes and polls the endpoint.
        Returns the endpoint URL on success, None on failure."""
        import time

        from huggingface_hub import get_inference_endpoint
        from huggingface_hub.utils import HfHubHTTPError

        try:
            ep = get_inference_endpoint(endpoint_name, namespace=self.namespace, token=self.token)
        except HfHubHTTPError:
            logger.error("HF endpoint '%s' not found — run provision script first", endpoint_name)
            return None

        status = ep.status
        logger.info("Endpoint '%s' status: %s", endpoint_name, status)

        if status == "running":
            return ep.url

        if status in ("paused", "scaledToZero"):
            logger.info("Waking endpoint '%s'...", endpoint_name)
            ep.resume()
        elif status == "initializing":
            logger.info("Endpoint '%s' already initializing, waiting...", endpoint_name)
        elif status == "failed":
            logger.error("Endpoint '%s' is in failed state", endpoint_name)
            return None

        # Poll until running or timeout
        deadline = time.monotonic() + WAKE_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL_SECONDS)
            try:
                ep = get_inference_endpoint(endpoint_name, namespace=self.namespace, token=self.token)
            except HfHubHTTPError:
                continue

            status = ep.status
            logger.info("Endpoint '%s' status: %s", endpoint_name, status)

            if status == "running":
                logger.info("Endpoint '%s' is ready at %s", endpoint_name, ep.url)
                return ep.url
            if status == "failed":
                logger.error("Endpoint '%s' entered failed state during wake-up", endpoint_name)
                return None

        logger.error("Endpoint '%s' did not become ready within %ds", endpoint_name, WAKE_TIMEOUT_SECONDS)
        return None
