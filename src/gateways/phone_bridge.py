import os
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# This bridge allows AnythingLLM (Docker) to request physical phone actions
# from David Thorne (the host-level Moto g06 via ADB).

app = FastAPI()

class Command(BaseModel):
    action: str # "screenshot" or "record"
    url: str | None = None

@app.post("/phone/execute")
async def execute_phone_action(cmd: Command):
    if cmd.action == "screenshot":
        # Capture screenshot
        subprocess.run(["adb", "shell", "input", "keyevent", "26"]) # Wake
        subprocess.run(["adb", "shell", "input", "swipe", "500", "1500", "500", "500"]) # Unlock
        subprocess.run(["adb", "exec-out", "screencap", "-p", ">", "/home/ubuntu/osia-framework/latest_intel.png"], shell=True)
        return {"status": "success", "message": "Screenshot captured on host."}
    
    return {"status": "error", "message": "Unknown action"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)
