import os
import subprocess
import logging
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import re

# This bridge allows AnythingLLM (Docker) to execute whitelisted cyber tools
# inside the Kali Sandbox (Docker) via host-level 'docker exec'.

logger = logging.getLogger("osia.cyber_bridge")
logging.basicConfig(level=logging.INFO)

CYBER_BRIDGE_TOKEN = os.getenv("CYBER_BRIDGE_TOKEN", "osia-cyber-secret-2026")
KALI_CONTAINER = "osia-kali"

app = FastAPI()
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != CYBER_BRIDGE_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

class CyberCommand(BaseModel):
    tool: str  # "nmap", "whois", "dig"
    target: str # IP or Domain

# Whitelist of allowed tools and their base arguments for safety
WHITELIST = {
    "nmap": ["nmap", "-T4", "-F"], # Fast scan of common ports
    "whois": ["whois"],
    "dig": ["dig", "+short"],
    "ping": ["ping", "-c", "4"]
}

def sanitize_input(text: str) -> bool:
    # Basic regex to allow only valid IPs or Domain names (prevent shell injection)
    pattern = r"^[a-zA-Z0-9\.-]+$"
    return bool(re.match(pattern, text))

@app.post("/cyber/execute")
async def execute_tool(cmd: CyberCommand, _=Depends(verify_token)):
    if cmd.tool not in WHITELIST:
        return {"status": "error", "message": f"Tool '{cmd.tool}' is not in the allowed whitelist."}

    if not sanitize_input(cmd.target):
        return {"status": "error", "message": "Invalid target. Only domains and IPs are allowed."}

    # Construct the command
    full_cmd = ["docker", "exec", KALI_CONTAINER] + WHITELIST[cmd.tool] + [cmd.target]
    
    logger.info("Executing cyber tool: %s", " ".join(full_cmd))
    
    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return {
                "status": "error", 
                "message": f"Tool execution failed.",
                "output": result.stderr
            }
        
        return {
            "status": "success",
            "tool": cmd.tool,
            "target": cmd.target,
            "output": result.stdout
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Tool execution timed out (60s limit)."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)
