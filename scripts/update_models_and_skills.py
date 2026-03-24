import os
import httpx
import logging
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).parent.parent / '.env')

ANYTHINGLLM_BASE_URL = os.getenv("ANYTHINGLLM_BASE_URL", "http://localhost:3001")
ANYTHINGLLM_API_KEY = os.getenv("ANYTHINGLLM_API_KEY")

if not ANYTHINGLLM_API_KEY:
    logger.error("ANYTHINGLLM_API_KEY is not set.")
    exit(1)

# List of global skills to ensure are activated
GLOBAL_SKILLS = [
    "web-browsing",
    "save-file-to-browser",
    "create-chart",
    "web-scraping",
    "osia-cyber-ip-intel",
    "osia-finance-stock-intel",
    "osia-stash-writer"
]

def main():
    headers = {
        "Authorization": f"Bearer {ANYTHINGLLM_API_KEY}",
        "Content-Type": "application/json"
    }

    # 1. Update Global System Skills (Internal API method using update-env or preferences if needed)
    # Note: We manually injected these into the sqlite DB for the moment, but normally 
    # AnythingLLM auto-enables any custom skill where active: true in plugin.json.
    logger.info("Custom skills loaded. Since they have active: true, AnythingLLM automatically detects them on next agent boot.")

    # 2. Update Workspace Models
    workspaces_url = f"{ANYTHINGLLM_BASE_URL.rstrip('/')}/api/v1/workspaces"
    try:
        response = httpx.get(workspaces_url, headers=headers)
        response.raise_for_status()
        workspaces = response.json().get("workspaces", [])
    except Exception as e:
        logger.error(f"Failed to fetch workspaces: {e}")
        return

    # Define the intended models for each desk
    desk_configs = {
        "cyber-intelligence-and-warfare-desk": {
            "chatProvider": "anthropic",
            "chatModel": "claude-sonnet-4-6",
            "agentProvider": "anthropic",
            "agentModel": "claude-sonnet-4-6"
        },
        "geopolitical-and-security-desk": {
            "chatProvider": "gemini",
            "chatModel": "gemini-3-flash",
            "agentProvider": "gemini",
            "agentModel": "gemini-3-flash"
        },
        "finance-and-economics-directorate": {
            "chatProvider": "openai",
            "chatModel": "gpt-5.4-mini",
            "agentProvider": "openai",
            "agentModel": "gpt-5.4-mini"
        },
        "the-watch-floor": {
            "chatProvider": "gemini",
            "chatModel": "gemini-3.1-pro-preview",
            "agentProvider": "gemini",
            "agentModel": "gemini-3.1-pro-preview"
        }
    }

    for ws in workspaces:
        slug = ws.get("slug")
        if slug in desk_configs:
            config = desk_configs[slug]
            logger.info(f"Updating models for {slug}...")
            update_url = f"{ANYTHINGLLM_BASE_URL.rstrip('/')}/api/v1/workspace/{slug}/update"
            
            try:
                update_response = httpx.post(update_url, headers=headers, json=config)
                update_response.raise_for_status()
                logger.info(f"Successfully configured agents & models for {slug}")
            except Exception as e:
                logger.error(f"Failed to update {slug}: {e}")

if __name__ == "__main__":
    main()