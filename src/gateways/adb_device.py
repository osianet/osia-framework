import asyncio
import logging
import re
import subprocess
import xml.etree.ElementTree as ET

logger = logging.getLogger("osia.adb")


class ADBDevice:
    """Gateway for controlling physical Android devices over USB/ADB for OSINT collection."""

    def __init__(self, device_id: str = None, lock_check=None):
        self.device_id = device_id
        self._lock_check = lock_check  # optional async callable — called before every _run
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
        if self._lock_check:
            await self._lock_check()
        cmd = self._build_cmd(args)
        return await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True, **kwargs)

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
        # Spaces → %s (ADB input text convention).
        # Wrap in single quotes so that $, ", `, &, |, ;, <, > etc. are all safe.
        # The only character that breaks single-quoting is ' itself — escape it via '\''.
        text_nospaces = text.replace(" ", "%s")
        escaped_text = "'" + text_nospaces.replace("'", "'\\''") + "'"
        await self._run_checked(["shell", "input", "text", escaped_text])

    async def take_screenshot(self, save_path: str):
        if self._lock_check:
            await self._lock_check()
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

        await self._run_checked(["shell", "input", "swipe", "500", "1500", "500", "500", "200"])
        await asyncio.sleep(1)

    async def open_url(self, url: str):
        await self.wake_and_unlock()
        logger.info("Opening URL %s on device...", url)
        await self._run_checked(["shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", url])

    async def record_screen(self, remote_path: str = "/sdcard/screen_capture.mp4", time_limit: int = 30):
        logger.info("Recording screen for %d seconds...", time_limit)
        cmd = self._build_cmd(["shell", "screenrecord", "--time-limit", str(time_limit), remote_path])
        # Add a hard subprocess timeout so a disconnected device can't hang the pipeline.
        # screenrecord should exit on its own after time_limit seconds; give it 30s grace.
        subprocess_timeout = time_limit + 30
        try:
            await asyncio.to_thread(subprocess.run, cmd, timeout=subprocess_timeout)
        except subprocess.TimeoutExpired:
            logger.warning(
                "screenrecord did not exit within %ds (time_limit=%ds) — "
                "device may have disconnected. Continuing with partial capture.",
                subprocess_timeout,
                time_limit,
            )

    async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300):
        """Perform a swipe gesture between two points."""
        await self._run_checked(["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)])

    async def press_back(self):
        """Press the Android back button."""
        await self._run_checked(["shell", "input", "keyevent", "4"])

    async def press_enter(self):
        """Press the Enter/Return key."""
        await self._run_checked(["shell", "input", "keyevent", "66"])

    async def press_send(self):
        """Simulate the Send/Submit action (e.g. keyboard send button)."""
        # Using KEYCODE_NUMPAD_ENTER (160) often triggers the IME's editor action (like Send)
        # on multiline fields where a regular Enter (66) would just insert a newline.
        # Alternatively, apps like Instagram support Ctrl+Enter via keycombination.
        # We try keycombination 113 66 (Ctrl+Enter) first, then fallback to 160 if needed.
        # Android 15 supports keycombination, so let's use Ctrl+Enter.
        await self._run_checked(["shell", "input", "keycombination", "113", "66"])

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

    async def dump_ui_tree(self) -> list[dict]:
        """
        Dump the current UI accessibility tree via uiautomator and return a flat
        list of all nodes that have non-empty bounds.

        Each node dict contains:
            text, content_desc, resource_id, class_name,
            clickable, bounds (raw string), cx, cy (center coords)
        """
        # Stream the XML directly — avoids a pull step and sdcard permission issues
        result = await self._run(["exec-out", "uiautomator", "dump", "/dev/tty"])
        xml_text = result.stdout.strip()
        if not xml_text:
            logger.warning("uiautomator dump returned empty output")
            return []

        # Strip any trailing garbage after the closing tag
        end = xml_text.rfind(">")
        if end != -1:
            xml_text = xml_text[: end + 1]

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.warning("Failed to parse uiautomator XML: %s", e)
            return []

        nodes = []
        for node in root.iter("node"):
            bounds_str = node.get("bounds", "")
            # bounds format: [x1,y1][x2,y2]
            m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
            if not m:
                continue
            x1, y1, x2, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                continue  # invisible / unmeasured element
            nodes.append(
                {
                    "text": node.get("text", ""),
                    "content_desc": node.get("content-desc", ""),
                    "resource_id": node.get("resource-id", ""),
                    "class_name": node.get("class", ""),
                    "clickable": node.get("clickable", "false") == "true",
                    "bounds": bounds_str,
                    "cx": (x1 + x2) // 2,
                    "cy": (y1 + y2) // 2,
                }
            )
        return nodes

    def find_element(
        self,
        nodes: list[dict],
        *,
        content_desc: str | None = None,
        resource_id: str | None = None,
        text: str | None = None,
        clickable_only: bool = True,
        fuzzy: bool = True,
    ) -> dict | None:
        """
        Search a node list (from dump_ui_tree) for the first matching element.

        Matching is case-insensitive. With fuzzy=True, a substring match is used;
        with fuzzy=False, an exact match is required.
        """

        def _match(value: str, target: str) -> bool:
            v, t = value.lower(), target.lower()
            return t in v if fuzzy else v == t

        for node in nodes:
            if clickable_only and not node["clickable"]:
                continue
            if content_desc and _match(node["content_desc"], content_desc):
                return node
            if resource_id and _match(node["resource_id"], resource_id):
                return node
            if text and _match(node["text"], text):
                return node
        return None

    async def tap_element(
        self,
        nodes: list[dict],
        *,
        content_desc: str | None = None,
        resource_id: str | None = None,
        text: str | None = None,
        clickable_only: bool = True,
        fuzzy: bool = True,
    ) -> bool:
        """
        Find an element in the UI tree and tap its center.
        Returns True if found and tapped, False otherwise.
        """
        node = self.find_element(
            nodes,
            content_desc=content_desc,
            resource_id=resource_id,
            text=text,
            clickable_only=clickable_only,
            fuzzy=fuzzy,
        )
        if node is None:
            return False
        logger.debug(
            "Tapping element '%s' / '%s' at (%d, %d)",
            node["content_desc"],
            node["resource_id"],
            node["cx"],
            node["cy"],
        )
        await self.tap(node["cx"], node["cy"])
        return True

    async def pull_file(self, remote_path: str, local_path: str):
        logger.info("Pulling %s to %s...", remote_path, local_path)
        await self._run_checked(["pull", remote_path, local_path])
