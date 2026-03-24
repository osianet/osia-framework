import asyncio
import logging
import re
import subprocess

logger = logging.getLogger("osia.adb")


class ADBDevice:
    """Gateway for controlling physical Android devices over USB/ADB for OSINT collection."""

    def __init__(self, device_id: str = None):
        self.device_id = device_id
        self._ensure_adb_started()

    def _ensure_adb_started(self):
        subprocess.run(
            ["adb", "start-server"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _build_cmd(self, args: list[str]) -> list[str]:
        cmd = ["adb"]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        cmd.extend(args)
        return cmd

    async def _run(self, args: list[str], **kwargs) -> subprocess.CompletedProcess:
        """Run an ADB command in a thread so we don't block the event loop."""
        cmd = self._build_cmd(args)
        return await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, **kwargs
        )

    async def _run_checked(self, args: list[str]) -> str:
        result = await self._run(args)
        if result.returncode != 0:
            raise RuntimeError(f"ADB command failed: {result.stderr}")
        return result.stdout.strip()

    # --- Sync helpers kept for non-async callers (e.g. __init__) ---

    def _run_adb_command(self, args: list[str]) -> str:
        cmd = self._build_cmd(args)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ADB Command Failed: {result.stderr}")
        return result.stdout.strip()

    def get_devices(self) -> list[str]:
        """Returns a list of connected ADB device IDs."""
        output = subprocess.run(["adb", "devices"], capture_output=True, text=True).stdout
        devices = []
        for line in output.split("\n")[1:]:
            if "\t" in line:
                device_id, state = line.split("\t")
                if state == "device":
                    devices.append(device_id)
        return devices

    async def tap(self, x: int, y: int):
        await self._run_checked(["shell", "input", "tap", str(x), str(y)])

    async def type_text(self, text: str):
        escaped_text = text.replace(" ", "%s")
        await self._run_checked(["shell", "input", "text", escaped_text])

    async def take_screenshot(self, save_path: str):
        cmd = self._build_cmd(["exec-out", "screencap", "-p"])
        result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"Screenshot failed: {result.stderr.decode()}")
        with open(save_path, "wb") as f:
            f.write(result.stdout)

    async def wake_and_unlock(self):
        logger.info("Waking and unlocking device...")
        # Check display state — split into two args so we avoid shell=True
        result = await self._run(["shell", "dumpsys", "power"])
        power_state = result.stdout

        if "mHoldingDisplaySuspendBlocker=false" in power_state:
            await self._run_checked(["shell", "input", "keyevent", "26"])
            await asyncio.sleep(1)

        await self._run_checked(
            ["shell", "input", "swipe", "500", "1500", "500", "500", "200"]
        )
        await asyncio.sleep(1)

    async def open_url(self, url: str):
        await self.wake_and_unlock()
        logger.info("Opening URL %s on device...", url)
        await self._run_checked(
            ["shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", url]
        )

    async def record_screen(
        self, remote_path: str = "/sdcard/screen_capture.mp4", time_limit: int = 30
    ):
        await self.wake_and_unlock()
        logger.info("Recording screen for %d seconds...", time_limit)
        cmd = self._build_cmd(
            ["shell", "screenrecord", "--time-limit", str(time_limit), remote_path]
        )
        await asyncio.to_thread(subprocess.run, cmd)

    async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300):
        """Perform a swipe gesture between two points."""
        await self._run_checked(
            ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)]
        )

    async def press_back(self):
        """Press the Android back button."""
        await self._run_checked(["shell", "input", "keyevent", "4"])

    async def press_enter(self):
        """Press the Enter/Return key."""
        await self._run_checked(["shell", "input", "keyevent", "66"])

    async def get_screen_size(self) -> tuple[int, int]:
        """Return (width, height) of the device display."""
        output = await self._run_checked(["shell", "wm", "size"])
        # Output looks like: "Physical size: 1080x2400"
        match = re.search(r"(\d+)x(\d+)", output)
        if not match:
            raise RuntimeError(f"Could not parse screen size from: {output}")
        return int(match.group(1)), int(match.group(2))

    async def get_foreground_app(self) -> str:
        """Return the package name of the currently focused app."""
        output = await self._run_checked(["shell", "dumpsys", "activity", "recents"])
        # Look for the top ResumedActivity line
        for line in output.splitlines():
            if "ResumedActivity" in line or "topResumedActivity" in line:
                # Format: ...ActivityRecord{... pkg/activity ...}
                match = re.search(r"([a-zA-Z][a-zA-Z0-9_.]+)/[a-zA-Z0-9_.]+", line)
                if match:
                    return match.group(1)
        # Fallback: mFocusedApp
        match = re.search(r"mFocusedApp.*?([a-zA-Z][a-zA-Z0-9_.]+)/", output)
        if match:
            return match.group(1)
        return ""

    async def pull_file(self, remote_path: str, local_path: str):
        logger.info("Pulling %s to %s...", remote_path, local_path)
        await self._run_checked(["pull", remote_path, local_path])
