import logging
import os
import httpx

logger = logging.getLogger("osia.anythingllm")


class AnythingLLMDesk:
    """Client for interacting with specialized AnythingLLM Workspaces (Desks)."""

    def __init__(self):
        self.base_url = os.getenv("ANYTHINGLLM_BASE_URL", "http://localhost:3001")
        self.api_key = os.getenv("ANYTHINGLLM_API_KEY", "")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # Shared client — avoids socket churn on repeated calls
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=300.0,
        )

    async def close(self):
        await self._client.aclose()

    async def send_task(self, workspace_slug: str, message: str) -> str:
        """Sends a query to a specific workspace and returns the response."""
        url = f"/api/v1/workspace/{workspace_slug}/chat"
        
        # Automatically trigger agent mode so AnythingLLM desks can use custom skills
        if not message.strip().startswith("@agent"):
            message = f"@agent {message}"
            
        payload = {"message": message, "mode": "chat"}

        response = await self._client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        text = data.get("textResponse") or ""
        if not text:
            logger.warning("Desk '%s' returned empty textResponse. Full payload: %s", workspace_slug, data)
            raise ValueError(f"Desk '{workspace_slug}' returned an empty response.")

        # AnythingLLM returns agent errors as textResponse — detect and raise
        agent_error_markers = [
            "The agent model failed to respond",
            "agent could not complete",
            "failed to generate a response",
        ]
        if any(marker.lower() in text.lower() for marker in agent_error_markers):
            logger.error("Desk '%s' agent error: %s", workspace_slug, text[:200])
            raise RuntimeError(f"Desk '{workspace_slug}' agent failed: {text[:200]}")

        return text

    async def ingest_raw_data(self, workspace_slug: str, text_content: str, title: str):
        """Dumps raw OSINT collection data into a workspace's Vector DB."""
        url = "/api/v1/document/raw-text"
        payload = {
            "textContent": text_content,
            "addToWorkspaces": workspace_slug,
            "metadata": {"title": title},
        }

        response = await self._client.post(url, json=payload, timeout=60.0)
        response.raise_for_status()
        return response.json()
