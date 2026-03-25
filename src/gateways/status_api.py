"""
OSIA Status API — exposes service health, metrics, and logs over HTTP.

Authentication:
  - Bearer token (STATUS_API_TOKEN env var)
  - User-Agent must contain the cryptic sentinel (STATUS_API_UA_SENTINEL env var)

Token management:
  uv run python scripts/manage_status_token.py          # show current token
  uv run python scripts/manage_status_token.py --rotate  # generate and save a new token
"""

import os
import json
import asyncio
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

load_dotenv()

logger = logging.getLogger("osia.status_api")

BASE_DIR = Path(os.getenv("OSIA_BASE_DIR", Path(__file__).resolve().parent.parent.parent))
STATUS_API_TOKEN = os.getenv("STATUS_API_TOKEN", "")
STATUS_API_UA_SENTINEL = os.getenv("STATUS_API_UA_SENTINEL", "osia-monitor/1")
STATUS_API_PORT = int(os.getenv("STATUS_API_PORT", "8099"))

SERVICES = [
    "osia-orchestrator.service",
    "osia-signal-ingress.service",
    "osia-persona-daemon.service",
    "osia-rss-ingress.service",
    "osia-mcp-arxiv-bridge.service",
    "osia-mcp-phone-bridge.service",
    "osia-mcp-semantic-scholar-bridge.service",
    "osia-mcp-tavily-bridge.service",
    "osia-mcp-time-bridge.service",
    "osia-mcp-wikipedia-bridge.service",
    "osia-cyber-bridge.service",
]

TIMERS = [
    "osia-daily-sitrep.timer",
    "osia-rss-ingress.timer",
]

CONTAINERS = [
    "osia-anythingllm",
    "osia-qdrant",
    "osia-redis",
    "osia-signal",
    "mailserver",
    "osia-kali",
]

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
security = HTTPBearer()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _check_auth(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not STATUS_API_TOKEN:
        raise HTTPException(status_code=503, detail="Status API not configured")
    ua = request.headers.get("user-agent", "")
    if STATUS_API_UA_SENTINEL not in ua:
        raise HTTPException(status_code=404, detail="Not found")
    if credentials.credentials != STATUS_API_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")


# ---------------------------------------------------------------------------
# Helpers — all run in a thread pool to avoid blocking the event loop
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 5) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except FileNotFoundError:
        return -1, "", "command not found"


def _systemd_service_info(name: str) -> dict:
    rc, out, _ = _run(["systemctl", "show", name,
                        "--property=ActiveState,SubState,MainPID,MemoryCurrent,ExecMainStartTimestamp"])
    props: dict = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            props[k] = v

    active = props.get("ActiveState", "unknown")
    sub = props.get("SubState", "unknown")
    pid = props.get("MainPID", "0")
    mem_bytes = props.get("MemoryCurrent", "")
    started = props.get("ExecMainStartTimestamp", "")

    mem_mb: float | None = None
    try:
        mem_mb = round(int(mem_bytes) / 1024 / 1024, 1)
    except (ValueError, TypeError):
        pass

    return {
        "name": name,
        "active": active,
        "sub": sub,
        "pid": pid if pid != "0" else None,
        "memory_mb": mem_mb,
        "started": started or None,
        "ok": active == "active",
    }


def _systemd_timer_info(name: str) -> dict:
    rc, out, _ = _run(["systemctl", "show", name,
                        "--property=ActiveState,NextElapseUSecRealtime,LastTriggerUSec"])
    props: dict = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            props[k] = v

    return {
        "name": name,
        "active": props.get("ActiveState", "unknown"),
        "next_elapse": props.get("NextElapseUSecRealtime", None),
        "last_trigger": props.get("LastTriggerUSec", None),
        "ok": props.get("ActiveState") == "active",
    }


def _docker_container_info(name: str) -> dict:
    rc, out, _ = _run(["docker", "inspect", "-f",
                        "{{.State.Status}}|{{.State.StartedAt}}|{{.State.Health.Status}}",
                        name])
    if rc != 0 or not out:
        return {"name": name, "status": "not found", "ok": False}
    parts = out.split("|")
    status = parts[0] if len(parts) > 0 else "unknown"
    started = parts[1] if len(parts) > 1 else None
    health = parts[2] if len(parts) > 2 else None
    return {
        "name": name,
        "status": status,
        "started": started,
        "health": health if health and health != "<no value>" else None,
        "ok": status == "running",
    }


def _system_metrics() -> dict:
    metrics: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": "",
        "uptime": "",
        "load": [],
        "memory": {},
        "disk": {},
        "cpu_temp_c": None,
        "gpu": None,
    }

    # hostname
    rc, out, _ = _run(["hostname"])
    metrics["hostname"] = out

    # uptime + load
    try:
        with open("/proc/uptime") as f:
            secs = float(f.read().split()[0])
            h, rem = divmod(int(secs), 3600)
            m = rem // 60
            metrics["uptime"] = f"{h}h {m}m"
    except Exception:
        pass

    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
            metrics["load"] = [float(parts[0]), float(parts[1]), float(parts[2])]
    except Exception:
        pass

    # memory
    try:
        rc, out, _ = _run(["free", "-m"])
        for line in out.splitlines():
            if line.startswith("Mem:"):
                parts = line.split()
                total, used, free = int(parts[1]), int(parts[2]), int(parts[3])
                metrics["memory"] = {
                    "total_mb": total,
                    "used_mb": used,
                    "free_mb": free,
                    "pct": round(used * 100 / total, 1) if total else 0,
                }
    except Exception:
        pass

    # disk
    try:
        rc, out, _ = _run(["df", "-m", str(BASE_DIR)])
        lines = out.splitlines()
        if len(lines) >= 2:
            parts = lines[1].split()
            total, used, avail = int(parts[1]), int(parts[2]), int(parts[3])
            metrics["disk"] = {
                "total_mb": total,
                "used_mb": used,
                "avail_mb": avail,
                "pct": round(used * 100 / total, 1) if total else 0,
            }
    except Exception:
        pass

    # CPU temp (ARM SBC)
    try:
        temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if temp_path.exists():
            metrics["cpu_temp_c"] = round(int(temp_path.read_text().strip()) / 1000, 1)
    except Exception:
        pass

    # GPU (nvidia-smi)
    try:
        rc, out, _ = _run(["nvidia-smi",
                            "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"])
        if rc == 0 and out:
            parts = [p.strip() for p in out.split(",")]
            metrics["gpu"] = {
                "name": parts[0],
                "temp_c": int(parts[1]),
                "util_pct": int(parts[2]),
                "vram_used_mb": int(parts[3]),
                "vram_total_mb": int(parts[4]),
            }
    except Exception:
        pass

    return metrics


def _redis_info() -> dict:
    rc, out, _ = _run(["docker", "exec", "osia-redis", "redis-cli", "ping"])
    if "PONG" not in out:
        return {"ok": False, "queue_depth": None}

    rc2, depth, _ = _run(["docker", "exec", "osia-redis", "redis-cli", "LLEN", "osia:task_queue"])
    rc3, peek, _ = _run(["docker", "exec", "osia-redis", "redis-cli", "LINDEX", "osia:task_queue", "0"])

    queue_depth = int(depth) if depth.isdigit() else 0
    return {
        "ok": True,
        "queue_depth": queue_depth,
        "next_task_preview": peek[:300] if peek else None,
    }


def _get_logs(service: str, lines: int = 100) -> list[str]:
    rc, out, err = _run(
        ["journalctl", "-u", service, "-n", str(lines), "--no-pager", "--output=short-iso"],
        timeout=10,
    )
    if rc != 0:
        return [f"ERROR: {err or 'journalctl failed'}"]
    return out.splitlines()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/status")
async def get_status(_=Depends(_check_auth)):
    """Full system status snapshot."""
    (
        system,
        services,
        timers,
        containers,
        redis_info,
    ) = await asyncio.gather(
        asyncio.to_thread(_system_metrics),
        asyncio.gather(*[asyncio.to_thread(_systemd_service_info, s) for s in SERVICES]),
        asyncio.gather(*[asyncio.to_thread(_systemd_timer_info, t) for t in TIMERS]),
        asyncio.gather(*[asyncio.to_thread(_docker_container_info, c) for c in CONTAINERS]),
        asyncio.to_thread(_redis_info),
    )

    return {
        "system": system,
        "services": list(services),
        "timers": list(timers),
        "containers": list(containers),
        "redis": redis_info,
    }


@app.get("/status/services")
async def get_services(_=Depends(_check_auth)):
    """Systemd service states only."""
    results = await asyncio.gather(*[asyncio.to_thread(_systemd_service_info, s) for s in SERVICES])
    return {"services": list(results)}


@app.get("/status/containers")
async def get_containers(_=Depends(_check_auth)):
    """Docker container states only."""
    results = await asyncio.gather(*[asyncio.to_thread(_docker_container_info, c) for c in CONTAINERS])
    return {"containers": list(results)}


@app.get("/status/system")
async def get_system(_=Depends(_check_auth)):
    """Host system metrics (CPU, memory, disk, GPU)."""
    return await asyncio.to_thread(_system_metrics)


@app.get("/status/redis")
async def get_redis(_=Depends(_check_auth)):
    """Redis health and task queue depth."""
    return await asyncio.to_thread(_redis_info)


@app.get("/logs/{service}")
async def get_service_logs(service: str, lines: int = 100, _=Depends(_check_auth)):
    """
    Tail logs for a named systemd service.
    Service name may omit the .service suffix.
    Max 500 lines per request.
    """
    if not service.endswith(".service"):
        service = f"{service}.service"
    # Whitelist — only OSIA services
    if service not in SERVICES:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service}")
    lines = min(lines, 500)
    log_lines = await asyncio.to_thread(_get_logs, service, lines)
    return {"service": service, "lines": log_lines, "count": len(log_lines)}


@app.get("/health")
async def health():
    """Unauthenticated liveness probe."""
    return {"ok": True}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=STATUS_API_PORT)
