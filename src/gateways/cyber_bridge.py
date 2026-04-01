import logging
import os
import re
import subprocess

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

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
    target: str  # IP or Domain


# Whitelist of allowed tools and their base arguments for safety
WHITELIST = {
    "nmap": ["nmap", "-T4", "-F"],  # Fast scan of common ports
    "whois": ["whois"],
    "dig": ["dig", "+short"],
    "ping": ["ping", "-c", "4"],
}


def sanitize_input(text: str) -> bool:
    # Basic regex to allow only valid IPs or Domain names (prevent shell injection)
    pattern = r"^[a-zA-Z0-9\.-]+$"
    return bool(re.match(pattern, text))


@app.post("/cyber/execute")
async def execute_tool(cmd: CyberCommand, _=Depends(verify_token)):
    if cmd.tool not in WHITELIST:
        return {"status": "error", "message": f"Tool '{cmd.tool}' is not in the allowed whitelist."}

    # Resolve to the whitelisted key to break CodeQL taint from user input
    tool_name = next(k for k in WHITELIST if k == cmd.tool)

    if not sanitize_input(cmd.target):
        return {"status": "error", "message": "Invalid target. Only domains and IPs are allowed."}

    # Construct the command — cmd.target is validated by sanitize_input() above
    # (only allows [a-zA-Z0-9.-]+) and tool_name is resolved from WHITELIST keys.
    full_cmd = ["docker", "exec", KALI_CONTAINER] + WHITELIST[tool_name] + [cmd.target]

    # Log only the whitelisted tool name — target is user-provided and excluded
    # from log output to prevent log injection (CWE-117)
    logger.info("Executing cyber tool: %s", tool_name)

    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return {"status": "error", "message": "Tool execution failed.", "output": result.stderr}

        return {"status": "success", "tool": tool_name, "target": cmd.target, "output": result.stdout}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Tool execution timed out (60s limit)."}
    except Exception:
        logger.exception("Cyber tool execution error")
        return {"status": "error", "message": "Internal execution error"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)
