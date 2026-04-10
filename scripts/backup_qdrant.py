#!/usr/bin/env python3
"""
Qdrant backup script — creates a full-instance snapshot via the REST API,
moves it to a dated directory, and prunes snapshots older than KEEP_DAYS.

Storage layout:
  /home/ubuntu/osia-qdrant-backups/
    2026-04-10/
      full-2026-04-10T02:00:00.snapshot
    2026-04-09/
      full-2026-04-09T02:00:00.snapshot
    ...

Run:  uv run python scripts/backup_qdrant.py
Env:  QDRANT_URL, QDRANT_API_KEY (from .env)
"""

import logging
import os
import shutil
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("osia.qdrant_backup")

# Always hit the local port directly — bypasses nginx proxy timeouts
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
# Full-instance snapshots go to /qdrant/snapshots inside the container,
# mounted here so files persist across container restarts
SNAPSHOT_SOURCE_DIR = Path("/home/ubuntu/osia-qdrant/snapshots")
BACKUP_BASE_DIR = Path("/home/ubuntu/osia-qdrant-backups")
KEEP_DAYS = 7
POLL_INTERVAL = 5   # seconds between status checks
TIMEOUT_SECS = 3600  # 1 hour max for snapshot creation


def headers() -> dict:
    return {"api-key": QDRANT_API_KEY} if QDRANT_API_KEY else {}


def create_snapshot(client: httpx.Client) -> str:
    """Trigger a full-instance snapshot and return the snapshot file name."""
    log.info("Requesting full Qdrant snapshot …")
    # POST /snapshots is synchronous — blocks until the snapshot file is written.
    # Allow up to 10 minutes for large instances.
    resp = client.post("/snapshots", timeout=600)
    resp.raise_for_status()
    data = resp.json()

    # Response: {"result": {"name": "...", "creation_time": ..., "size": ...}, "status": "ok"}
    name = data["result"]["name"]
    log.info("Snapshot created: %s", name)
    return name


def wait_for_snapshot(name: str) -> None:
    """Verify the snapshot file exists — POST /snapshots is synchronous so it should be ready immediately."""
    snap_path = SNAPSHOT_SOURCE_DIR / name
    if snap_path.exists() and snap_path.stat().st_size > 0:
        log.info("Snapshot file ready: %s (%.1f MB)", snap_path, snap_path.stat().st_size / 1e6)
        return
    # Fallback: brief poll in case of minor filesystem lag
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if snap_path.exists() and snap_path.stat().st_size > 0:
            log.info("Snapshot file ready: %s (%.1f MB)", snap_path, snap_path.stat().st_size / 1e6)
            return
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Snapshot {name} not found at {snap_path} after 60s")


def move_to_backup(name: str) -> Path:
    """Move the snapshot from the Qdrant storage dir to the dated backup dir."""
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    dest_dir = BACKUP_BASE_DIR / today
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = SNAPSHOT_SOURCE_DIR / name
    dst = dest_dir / name
    shutil.move(str(src), str(dst))
    log.info("Snapshot archived to: %s", dst)
    return dst


def prune_old_backups() -> None:
    """Delete backup directories older than KEEP_DAYS."""
    if not BACKUP_BASE_DIR.exists():
        return
    cutoff = datetime.now(UTC) - timedelta(days=KEEP_DAYS)
    for entry in sorted(BACKUP_BASE_DIR.iterdir()):
        if not entry.is_dir():
            continue
        try:
            entry_date = datetime.strptime(entry.name, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            continue
        if entry_date < cutoff:
            shutil.rmtree(entry)
            log.info("Pruned old backup: %s", entry)


def delete_qdrant_snapshot_record(client: httpx.Client, name: str) -> None:
    """Remove the snapshot record from Qdrant's internal list (file already moved)."""
    try:
        resp = client.delete(f"/snapshots/{name}", timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("Could not delete snapshot record from Qdrant: %s", exc)


def main() -> None:
    BACKUP_BASE_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(base_url=QDRANT_URL, headers=headers()) as client:
        # Verify Qdrant is reachable
        try:
            client.get("/", timeout=10).raise_for_status()
        except Exception as exc:
            log.error("Cannot reach Qdrant at %s: %s", QDRANT_URL, exc)
            sys.exit(1)

        name = create_snapshot(client)
        wait_for_snapshot(name)
        archived = move_to_backup(name)
        delete_qdrant_snapshot_record(client, name)

    prune_old_backups()

    size_mb = archived.stat().st_size / 1e6
    log.info("Backup complete — %.1f MB saved to %s", size_mb, archived)


if __name__ == "__main__":
    main()
