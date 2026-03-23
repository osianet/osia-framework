import os
import httpx

class AnythingLLMDesk:
    """Client for interacting with specialized AnythingLLM Workspaces (Desks)."""
    
    def __init__(self):
        self.base_url = os.getenv("ANYTHINGLLM_BASE_URL", "http://localhost:3001")
        self.api_key = os.getenv("ANYTHINGLLM_API_KEY", "")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    async def send_task(self, workspace_slug: str, message: str) -> str:
        """Sends a query to a specific workspace and returns the response."""
        url = f"{self.base_url}/api/v1/workspace/{workspace_slug}/chat"
        
        # Define model overrides for specialized desks
        model_overrides = {
            "finance-and-economics-directorate": "gpt-4o",
            "geopolitical-and-security-desk": "gemini-2.5-flash",
            "cultural-and-theological-intelligence-desk": "gemini-2.5-flash",
            "the-watch-floor": "gemini-2.5-flash"
        }

        # We attach the collection-directorate documents to every query 
        # so the specialized desk can "read" the latest raw intel.
        payload = {
            "message": message,
            "mode": "chat",
            "attachments": [
                {
                    "workspaceSlug": "collection-directorate"
                }
            ]
        }

        if workspace_slug in model_overrides:
            # Note: AnythingLLM API might need model custom settings updated
            # For now we use the system defaults but ensure the slug is correct.
            pass
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("textResponse", "")

    async def ingest_raw_data(self, workspace_slug: str, text_content: str, title: str):
        """Dumps raw OSINT collection data into a workspace's Vector DB."""
        url = f"{self.base_url}/api/v1/document/raw-text"
        payload = {
            "textContent": text_content,
            "addToWorkspaces": workspace_slug,
            "metadata": {
                "title": title
            }
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
