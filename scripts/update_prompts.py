import os
import json
import httpx
import logging
from pathlib import Path
from dotenv import load_dotenv
from google import genai

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv(Path(__file__).parent.parent / '.env')

ANYTHINGLLM_BASE_URL = os.getenv("ANYTHINGLLM_BASE_URL", "http://localhost:3001")
ANYTHINGLLM_API_KEY = os.getenv("ANYTHINGLLM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

if not ANYTHINGLLM_API_KEY:
    logger.error("ANYTHINGLLM_API_KEY is not set.")
    exit(1)

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is not set.")
    exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)

BASE_DIR = Path(__file__).parent.parent
DIRECTIVES_PATH = BASE_DIR / "DIRECTIVES.md"
TEMPLATES_DIR = BASE_DIR / "templates" / "prompts"

def generate_prompt(template: str, directives: str) -> str:
    prompt = f"""You are an expert system prompt engineer for an AI system called OSIA. 
I will provide you with a base template for a specific AI agent desk, and the core organizational directives. 
Your job is to create a unified, clear, and comprehensive system prompt.

Rules:
1. Keep the prompt focused on the agent's specific role as defined in the base template.
2. Incorporate the spirit and rules of the DIRECTIVES naturally into the agent's instructions.
3. PRESERVE VERBATIM any instructions marked as "MANDATORY" from the base template (e.g., fetching time, formatting). Do not change them.
4. Output ONLY the final system prompt text. Do not include any introductory or explanatory text.

=== BASE TEMPLATE ===
{template}

=== CORE DIRECTIVES ===
{directives}
"""
    response = client.models.generate_content(
        model=GEMINI_MODEL_ID,
        contents=prompt,
    )
    return response.text.strip()

def main():
    if not DIRECTIVES_PATH.exists():
        logger.error(f"Directives file not found at {DIRECTIVES_PATH}")
        return

    directives = DIRECTIVES_PATH.read_text()
    
    headers = {
        "Authorization": f"Bearer {ANYTHINGLLM_API_KEY}",
        "Content-Type": "application/json"
    }

    # Fetch workspaces
    workspaces_url = f"{ANYTHINGLLM_BASE_URL.rstrip('/')}/api/v1/workspaces"
    try:
        response = httpx.get(workspaces_url, headers=headers)
        response.raise_for_status()
        workspaces = response.json().get("workspaces", [])
    except Exception as e:
        logger.error(f"Failed to fetch workspaces: {e}")
        return

    for ws in workspaces:
        slug = ws.get("slug")
        template_file = TEMPLATES_DIR / f"{slug}.txt"
        
        if not template_file.exists():
            logger.info(f"Skipping {slug} - no template found.")
            continue
            
        logger.info(f"Processing workspace: {slug}")
        template = template_file.read_text()
        
        try:
            new_prompt = generate_prompt(template, directives)
            logger.info(f"Generated new prompt for {slug} ({len(new_prompt)} chars)")
            
            update_url = f"{ANYTHINGLLM_BASE_URL.rstrip('/')}/api/v1/workspace/{slug}/update"
            update_data = {
                "openAiPrompt": new_prompt
            }
            update_response = httpx.post(update_url, headers=headers, json=update_data)
            update_response.raise_for_status()
            logger.info(f"Successfully updated AnythingLLM workspace: {slug}")
            
        except Exception as e:
            logger.error(f"Failed to update {slug}: {e}")

if __name__ == "__main__":
    main()