import asyncio
import base64
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import redis.asyncio as redis
import websockets
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("osia.signal_ingress")

SIGNAL_NUMBER = os.getenv("SIGNAL_SENDER_NUMBER")
if not SIGNAL_NUMBER:
    raise ValueError("SIGNAL_SENDER_NUMBER environment variable is required")

_signal_ws_base = os.getenv("SIGNAL_WS_URL", "ws://localhost:8081")
SIGNAL_WS_URL = f"{_signal_ws_base}/v1/receive/{SIGNAL_NUMBER}"
# Derive HTTP base URL from the websocket base (ws:// → http://, wss:// → https://)
SIGNAL_API_URL = os.getenv("SIGNAL_API_URL", _signal_ws_base.replace("wss://", "https://").replace("ws://", "http://"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TASK_QUEUE = os.getenv("OSIA_TASK_QUEUE", "osia:task_queue")
SIGNAL_CONTAINER_NAME = os.getenv("SIGNAL_CONTAINER_NAME", "osia-signal")
# Restart the signal-cli container if the WebSocket has been down for this long
_WS_RESTART_THRESHOLD_SECS = 120

# Temp directory for Signal attachment downloads (cleaned up after task is queued)
_ATTACH_DIR = Path(tempfile.gettempdir()) / "osia_signal_attachments"
_ATTACH_DIR.mkdir(parents=True, exist_ok=True)

_SUPPORTED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/webm",
}


async def _download_attachment(http: httpx.AsyncClient, attachment: dict) -> dict | None:
    """
    Download a single Signal attachment and save it to a temp file.

    Returns a dict with ``path`` and ``content_type``, or None on failure.
    """
    attachment_id = attachment.get("id")
    content_type = attachment.get("contentType", "")

    if not attachment_id or content_type not in _SUPPORTED_CONTENT_TYPES:
        return None

    try:
        resp = await http.get(f"{SIGNAL_API_URL}/v1/attachments/{attachment_id}", timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to download Signal attachment %s: %s", attachment_id, exc)
        return None

    # Derive a sensible file extension from the content type
    ext = content_type.split("/")[-1].replace("jpeg", "jpg").replace("quicktime", "mov")
    dest = _ATTACH_DIR / f"{attachment_id}.{ext}"
    dest.write_bytes(resp.content)
    logger.info("Downloaded Signal attachment %s → %s (%d bytes)", attachment_id, dest, len(resp.content))
    return {"path": str(dest), "content_type": content_type}


def _restart_signal_container() -> None:
    """Restart the signal-cli Docker container to recover from a broken internal state."""
    logger.warning(
        "WebSocket has been down for >%ds — restarting container %s",
        _WS_RESTART_THRESHOLD_SECS,
        SIGNAL_CONTAINER_NAME,
    )
    try:
        subprocess.run(
            ["docker", "restart", SIGNAL_CONTAINER_NAME],
            timeout=30,
            check=True,
            capture_output=True,
        )
        logger.warning("Container %s restarted successfully.", SIGNAL_CONTAINER_NAME)
    except Exception as exc:
        logger.error("Failed to restart container %s: %s", SIGNAL_CONTAINER_NAME, exc)


async def listen_to_signal():
    """Connects to the Signal REST API WebSocket and listens for incoming messages."""
    redis_client = redis.from_url(REDIS_URL)

    logger.info("Signal Gateway starting... Listening on %s", SIGNAL_WS_URL)

    failure_since: float | None = None

    while True:
        try:
            async with websockets.connect(SIGNAL_WS_URL) as websocket:
                logger.info("Connected to Signal WebSocket!")
                failure_since = None
                async with httpx.AsyncClient() as http:
                    while True:
                        message_str = await websocket.recv()
                        try:
                            data = json.loads(message_str)
                        except json.JSONDecodeError:
                            continue

                        if "envelope" not in data:
                            continue

                        envelope = data["envelope"]

                        # syncMessage envelopes are OSIA's own outbound messages echoed
                        # back for multi-device sync — ignore them entirely.
                        if "syncMessage" in envelope:
                            continue

                        if "dataMessage" not in envelope:
                            continue

                        source = envelope.get("source")
                        data_msg = envelope["dataMessage"]

                        # Resolve group ID for reply routing
                        group_id = None
                        group_info = data_msg.get("groupInfo")
                        if group_info:
                            raw_group_id = group_info.get("groupId")
                            if raw_group_id:
                                if raw_group_id.startswith("group."):
                                    group_id = raw_group_id
                                else:
                                    encoded_id = base64.b64encode(raw_group_id.encode()).decode()
                                    group_id = f"group.{encoded_id}"

                        text = data_msg.get("message") or ""

                        # Download any image/video attachments
                        raw_attachments = data_msg.get("attachments") or []
                        attachments = []
                        for att in raw_attachments:
                            result = await _download_attachment(http, att)
                            if result:
                                attachments.append(result)

                        # Skip if there's nothing to process
                        if not text and not attachments:
                            continue

                        task_source = f"signal:{group_id}" if group_id else f"signal:{source}"
                        task = {"source": task_source, "query": text}
                        if attachments:
                            task["attachments"] = attachments

                        logger.info(
                            "[Signal] Message from %s (group: %s) — text: %r, attachments: %d",
                            source,
                            group_id,
                            text[:80] if text else "",
                            len(attachments),
                        )
                        await redis_client.rpush(TASK_QUEUE, json.dumps(task))
                        logger.info("Task pushed to %s", TASK_QUEUE)

        except Exception as e:
            now = time.monotonic()
            if failure_since is None:
                failure_since = now
            elapsed = now - failure_since
            logger.warning(
                "WebSocket connection dropped: %s. Reconnecting in 5 seconds... (down %.0fs)",
                e,
                elapsed,
            )
            if elapsed >= _WS_RESTART_THRESHOLD_SECS:
                await asyncio.get_event_loop().run_in_executor(None, _restart_signal_container)
                failure_since = None
                await asyncio.sleep(15)
            else:
                await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(listen_to_signal())
