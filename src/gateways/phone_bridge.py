import os
import logging
import subprocess
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import asyncio
from src.gateways.adb_device import ADBDevice

# This bridge allows AnythingLLM (Docker) to request physical phone actions
# from the host-level Moto g06 via ADB.

logger = logging.getLogger("osia.phone_bridge")

BASE_DIR = Path(os.getenv("OSIA_BASE_DIR", Path(__file__).resolve().parent.parent.parent))
PHONE_BRIDGE_TOKEN = os.getenv("PHONE_BRIDGE_TOKEN", "")

app = FastAPI()
security = HTTPBearer()

# We can specify the ADB device ID in the .env file for the phone bridge
adb = ADBDevice(device_id=os.getenv("ADB_DEVICE_DEFAULT"))


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple bearer-token auth so the endpoint isn't wide open."""
    if not PHONE_BRIDGE_TOKEN:
        logger.warning("PHONE_BRIDGE_TOKEN is not set — all requests will be rejected.")
        raise HTTPException(status_code=503, detail="Bridge not configured")
    if credentials.credentials != PHONE_BRIDGE_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


@app.get("/health")
async def health_check():
    """Unauthenticated health check — reports bridge and phone status."""
    phone_connected = False
    device_id = adb.device_id
    try:
        devices = await asyncio.to_thread(adb.get_devices)
        if device_id:
            phone_connected = device_id in devices
        else:
            phone_connected = len(devices) > 0
            device_id = devices[0] if devices else None
    except Exception as e:
        logger.warning("Health check ADB probe failed: %s", e)

    return {
        "status": "ok",
        "phone_connected": phone_connected,
        "device_id": device_id,
        "bridge_configured": bool(PHONE_BRIDGE_TOKEN),
    }


class Command(BaseModel):
    action: str  # "screenshot" or "record"
    url: str | None = None


@app.post("/phone/execute")
async def execute_phone_action(cmd: Command, _=Depends(verify_token)):
    if cmd.action == "screenshot":
        screenshot_path = BASE_DIR / "latest_intel.png"
        try:
            # Wake + unlock + capture
            await adb.wake_and_unlock()
            await adb.take_screenshot(str(screenshot_path))
            return {"status": "success", "message": f"Screenshot saved to {screenshot_path}"}
        except RuntimeError as e:
            return {"status": "error", "message": f"screencap failed: {e}"}

    return {"status": "error", "message": "Unknown action"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)

