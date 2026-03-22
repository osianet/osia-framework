import subprocess
import time

class ADBDevice:
    """Gateway for controlling physical Android devices over USB/ADB for true OSINT collection."""
    
    def __init__(self, device_id: str = None):
        self.device_id = device_id
        self._ensure_adb_started()

    def _ensure_adb_started(self):
        subprocess.run(["adb", "start-server"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _run_adb_command(self, args: list) -> str:
        cmd = ["adb"]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        cmd.extend(args)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ADB Command Failed: {result.stderr}")
        return result.stdout.strip()

    def get_devices(self) -> list:
        """Returns a list of connected ADB device IDs."""
        output = subprocess.run(["adb", "devices"], capture_output=True, text=True).stdout
        devices = []
        for line in output.split('\n')[1:]:
            if '\t' in line:
                device_id, state = line.split('\t')
                if state == 'device':
                    devices.append(device_id)
        return devices

    def tap(self, x: int, y: int):
        """Simulates a physical screen tap."""
        self._run_adb_command(["shell", "input", "tap", str(x), str(y)])

    def type_text(self, text: str):
        """Types text into the currently focused input field."""
        # Escape spaces for ADB
        escaped_text = text.replace(" ", "%s")
        self._run_adb_command(["shell", "input", "text", escaped_text])

    def take_screenshot(self, save_path: str):
        """Captures the device screen and saves it locally."""
        cmd = ["adb"]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        cmd.extend(["exec-out", "screencap", "-p"])
        with open(save_path, "wb") as f:
            subprocess.run(cmd, stdout=f, check=True)

    def wake_and_unlock(self):
        """Wakes up the device and performs a swipe to bypass the lock screen."""
        print("[*] ADB: Waking and unlocking device...")
        # Use a single shell command string to allow piping
        cmd = ["adb"]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        cmd.extend(["shell", "dumpsys power | grep mHoldingDisplaySuspendBlocker"])
        result = subprocess.run(cmd, capture_output=True, text=True)
        power_state = result.stdout

        if "false" in power_state.lower():
            # Press power button
            self._run_adb_command(["shell", "input", "keyevent", "26"])
            time.sleep(1)
        
        # Swipe up to unlock (works for 'Swipe' or 'None' lock types)
        self._run_adb_command(["shell", "input", "swipe", "500", "1500", "500", "500", "200"])
        time.sleep(1)

    def open_url(self, url: str):
        """Opens a URL using the default Android intent viewer."""
        self.wake_and_unlock()
        print(f"[*] ADB: Opening URL {url} on device...")
        self._run_adb_command(["shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", url])

    def record_screen(self, remote_path: str = "/sdcard/screen_capture.mp4", time_limit: int = 30):
        """Records the device screen for a specified duration."""
        self.wake_and_unlock()
        print(f"[*] ADB: Recording screen for {time_limit} seconds...")
        # We use subprocess.run here without check=True since screenrecord might hit the time limit and exit cleanly
        cmd = ["adb"]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        cmd.extend(["shell", "screenrecord", "--time-limit", str(time_limit), remote_path])
        subprocess.run(cmd)

    def pull_file(self, remote_path: str, local_path: str):
        """Pulls a file from the device to the local filesystem."""
        print(f"[*] ADB: Pulling {remote_path} to {local_path}...")
        self._run_adb_command(["pull", remote_path, local_path])
