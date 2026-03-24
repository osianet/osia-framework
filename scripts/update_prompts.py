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

# Configuration mapping defining exactly which agent and chat models should run each desk
DESK_MODELS = {
    "cyber-intelligence-and-warfare-desk": {
        "chatProvider": "generic-openai",
        "chatModel": "Dolphin-3.0-70B",
        "agentProvider": "generic-openai",
        "agentModel": "Dolphin-3.0-70B",
        "vectorTag": "cyber_intel"
    } if os.getenv("HF_ENDPOINT_DOLPHIN_70B") else {
        "chatProvider": "anthropic",
        "chatModel": "claude-sonnet-4-6",
        "agentProvider": "anthropic",
        "agentModel": "claude-sonnet-4-6",
        "vectorTag": "cyber_intel"
    },
    "geopolitical-and-security-desk": {
        "chatProvider": "gemini",
        "chatModel": "gemini-2.5-flash",
        "agentProvider": "gemini",
        "agentModel": "gemini-2.5-flash",
        "vectorTag": "geopolitical_intel"
    },
    "cultural-and-theological-intelligence-desk": {
        "chatProvider": "generic-openai",
        "chatModel": "Dolphin-3.0-8B",
        "agentProvider": "generic-openai",
        "agentModel": "Dolphin-3.0-8B",
        "vectorTag": "cultural_intel"
    } if os.getenv("HF_ENDPOINT_DOLPHIN_8B") else {
        "chatProvider": "gemini",
        "chatModel": "gemini-2.5-flash",
        "agentProvider": "gemini",
        "agentModel": "gemini-2.5-flash",
        "vectorTag": "cultural_intel"
    },
    "science-technology-and-commercial-desk": {
        "chatProvider": "anthropic",
        "chatModel": "claude-sonnet-4-6",
        "agentProvider": "anthropic",
        "agentModel": "claude-sonnet-4-6",
        "vectorTag": "science_intel"
    },
    "human-intelligence-and-profiling-desk": {
        "chatProvider": "generic-openai",
        "chatModel": "Dolphin-3.0-70B",
        "agentProvider": "generic-openai",
        "agentModel": "Dolphin-3.0-70B",
        "vectorTag": "human_intel"
    } if os.getenv("HF_ENDPOINT_DOLPHIN_70B") else {
        "chatProvider": "ollama",
        "chatModel": "nchapman/dolphin3.0-llama3:latest",
        "agentProvider": "ollama",
        "agentModel": "nchapman/dolphin3.0-llama3:latest",
        "vectorTag": "human_intel"
    },
    "finance-and-economics-directorate": {
        "chatProvider": "openai",
        "chatModel": "gpt-4o",
        "agentProvider": "openai",
        "agentModel": "gpt-4o",
        "vectorTag": "finance_intel"
    },
    "the-watch-floor": {
        "chatProvider": "gemini",
        "chatModel": "gemini-2.5-pro",
        "agentProvider": "gemini",
        "agentModel": "gemini-2.5-pro",
        "vectorTag": "watch_floor"
    },
    "collection-directorate": {
        "chatProvider": "generic-openai",
        "chatModel": "Pleias-RAG-350M",
        "agentProvider": "generic-openai",
        "agentModel": "Pleias-RAG-350M",
        "vectorTag": "collection_raw"
    }
}

# Apply HF URLs if they exist
if os.getenv("HF_ENDPOINT_DOLPHIN_70B"):
    url = os.getenv("HF_ENDPOINT_DOLPHIN_70B").rstrip("/") + "/v1"
    DESK_MODELS["cyber-intelligence-and-warfare-desk"]["basePath"] = url
    DESK_MODELS["human-intelligence-and-profiling-desk"]["basePath"] = url

if os.getenv("HF_ENDPOINT_DOLPHIN_8B"):
    url = os.getenv("HF_ENDPOINT_DOLPHIN_8B").rstrip("/") + "/v1"
    DESK_MODELS["cultural-and-theological-intelligence-desk"]["basePath"] = url

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

def activate_global_skills():
    """Update AnythingLLM sqlite to ensure all custom skills are loaded into default_agent_skills"""
    import sqlite3
    db_path = "/home/ubuntu/osia-knowledge-base/anythingllm.db"
    
    if not os.path.exists(db_path):
        logger.warning("Could not find anythingllm.db to update global skills.")
        return
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # This exact JSON array defines the default enabled skills across all agents
        # We REMOVE "web-scraping" because AnythingLLM sometimes injects it twice, causing Claude tool errors.
        skills_json = '["web-browsing","save-file-to-browser","create-chart","osia-cyber-ip-intel","osia-finance-stock-intel","osia-stash-writer","osia-geopol-country-intel","osia-social-username-recon","osia-culture-observatory","osia-github-repo-intel","osia-report-broadcast","osia-cyber-kali-tools"]'
        
        cursor.execute("UPDATE system_settings SET value = ? WHERE label = 'default_agent_skills';", (skills_json,))
        
        # Explicitly disable web-scraping to ensure it doesn't appear twice
        cursor.execute("UPDATE system_settings SET value = '[\"web-scraping\"]' WHERE label = 'disabled_agent_skills';")
        
        conn.commit()
        conn.close()
        logger.info("Successfully activated all OSIA custom skills globally in the database.")
    except Exception as e:
        logger.error(f"Failed to activate custom skills in db: {e}")

def main():
    if not DIRECTIVES_PATH.exists():
        logger.error(f"Directives file not found at {DIRECTIVES_PATH}")
        return

    # First ensure the global skills are toggled on
    activate_global_skills()

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
        
        # Base update payload
        update_data = {}
        
        # Inject Model Configurations if we have them mapped
        if slug in DESK_MODELS:
            logger.info(f"Injecting model configurations for {slug}")
            update_data.update(DESK_MODELS[slug])
        
        # Generate new prompt if template exists
        if template_file.exists():
            logger.info(f"Processing prompt for workspace: {slug}")
            template = template_file.read_text()
            try:
                new_prompt = generate_prompt(template, directives)
                update_data["openAiPrompt"] = new_prompt
                logger.info(f"Generated new prompt for {slug} ({len(new_prompt)} chars)")
            except Exception as e:
                logger.error(f"Failed to generate prompt for {slug}: {e}")
        
        if not update_data:
            logger.info(f"Skipping {slug} - no updates to push.")
            continue
            
        # Push the unified update
        try:
            update_url = f"{ANYTHINGLLM_BASE_URL.rstrip('/')}/api/v1/workspace/{slug}/update"
            update_response = httpx.post(update_url, headers=headers, json=update_data)
            update_response.raise_for_status()
            logger.info(f"Successfully updated AnythingLLM workspace: {slug}")
            
        except Exception as e:
            logger.error(f"Failed to push updates for {slug}: {e}")

if __name__ == "__main__":
    main()