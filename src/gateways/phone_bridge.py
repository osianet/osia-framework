import os
import logging
import subprocess
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

# This bridge allows AnythingLLM (Docker) to request physical phone actions
# from the host-level Moto g06 via ADB.

logger = logging.getLogger("osia.phone_bridge")

BASE_DIR = Path(os.getenv("OSIA_BASE_DIR", Path(__file__).resolve().parent.parent.parent))
PHONE_BRIDGE_TOKEN = os.getenv("PHONE_BRIDGE_TOKEN", "")

app = FastAPI()
security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple bearer-token auth so the endpoint isn't wide open."""
    if not PHONE_BRIDGE_TOKEN:
        logger.warning("PHONE_BRIDGE_TOKEN is not set — all requests will be rejected.")
        raise HTTPException(status_code=503, detail="Bridge not configured")
    if credentials.credentials != PHONE_BRIDGE_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


class Command(BaseModel):
    action: str  # "screenshot" or "record"
    url: str | None = None


@app.post("/phone/execute")
async def execute_phone_action(cmd: Command, _=Depends(verify_token)):
    if cmd.action == "screenshot":
        screenshot_path = BASE_DIR / "latest_intel.png"
        # Wake + unlock
        subprocess.run(["adb", "shell", "input", "keyevent", "26"])
        subprocess.run(["adb", "shell", "input", "swipe", "500", "1500", "500", "500"])
        # Capture — no shell=True, write the bytes from stdout instead
        result = subprocess.run(
            ["adb", "exec-out", "screencap", "-p"],
            capture_output=True,
        )
        if result.returncode != 0:
            return {"status": "error", "message": f"screencap failed: {result.stderr.decode()}"}
        screenshot_path.write_bytes(result.stdout)
        return {"status": "success", "message": f"Screenshot saved to {screenshot_path}"}

    return {"status": "error", "message": "Unknown action"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)
