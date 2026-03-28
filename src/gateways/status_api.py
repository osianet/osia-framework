"""
OSIA Status API — exposes service health, metrics, and logs over HTTP.

Authentication:
  - Bearer token (STATUS_API_TOKEN env var) — compared with secrets.compare_digest
  - User-Agent must contain the cryptic sentinel (STATUS_API_UA_SENTINEL env var)
  - Wrong UA returns 404 (service appears non-existent to scanners)

Token management:
  uv run python scripts/manage_status_token.py          # show current token
  uv run python scripts/manage_status_token.py --rotate  # generate and save a new token

Security posture:
  - All subprocess calls use list form (no shell=True, no user input interpolated)
  - Service names validated against a hardcoded whitelist before any syscall
  - Log output sanitised — ANSI/control characters stripped
  - Redis task preview redacted to type/source fields only (no payload content)
  - Token comparison is constant-time (secrets.compare_digest)
  - Rate limiting: 30 req/min per IP via slowapi
  - Server header suppressed
  - No shell=True anywhere
"""

import asyncio
import hmac
import json
import logging
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

logger = logging.getLogger("osia.status_api")

BASE_DIR = Path(os.getenv("OSIA_BASE_DIR", Path(__file__).resolve().parent.parent.parent))
STATUS_API_TOKEN = os.getenv("STATUS_API_TOKEN", "")
STATUS_API_UA_SENTINEL = os.getenv("STATUS_API_UA_SENTINEL", "osia-monitor/1")
STATUS_API_PORT = int(os.getenv("STATUS_API_PORT", "8099"))

# Hardcoded whitelists — nothing outside these lists ever touches a subprocess
SERVICES: list[str] = [
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
    "osia-status-api.service",
    "osia-queue-api.service",
]

TIMERS: list[str] = [
    "osia-daily-sitrep.timer",
    "osia-rss-ingress.timer",
]

QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Known desk collections — mirrors the vectorTag assignments in ANYTHINGLLM_CONFIG.md
QDRANT_DESK_COLLECTIONS = [
    "collection-directorate",
    "geopolitical-and-security-desk",
    "cultural-and-theological-intelligence-desk",
    "science-technology-and-commercial-desk",
    "human-intelligence-and-profiling-desk",
    "finance-and-economics-directorate",
    "cyber-intelligence-and-warfare-desk",
    "the-watch-floor",
]

CONTAINERS: list[str] = [
    "osia-anythingllm",
    "osia-qdrant",
    "osia-redis",
    "osia-signal",
    "mailserver",
    "osia-kali",
]

# Pre-compiled pattern for stripping ANSI escape codes and non-printable
# control characters from log output before returning it to the caller.
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
security = HTTPBearer()


# ---------------------------------------------------------------------------
# Middleware — strip server fingerprint header
# ---------------------------------------------------------------------------


@app.middleware("http")
async def _remove_server_header(request: Request, call_next):
    response = await call_next(request)
    if "server" in response.headers:
        del response.headers["server"]
    if "x-powered-by" in response.headers:
        del response.headers["x-powered-by"]
    return response


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _check_auth(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Two-factor gate:
      1. User-Agent must contain the sentinel string (wrong UA → 404, looks like nothing is here)
      2. Bearer token must match via constant-time comparison (wrong token → 403)
    """
    if not STATUS_API_TOKEN:
        raise HTTPException(status_code=503, detail="Not configured")

    ua = request.headers.get("user-agent", "")
    if STATUS_API_UA_SENTINEL not in ua:
        raise HTTPException(status_code=404, detail="Not found")

    # Constant-time comparison — prevents timing oracle on the token
    if not hmac.compare_digest(
        credentials.credentials.encode(),
        STATUS_API_TOKEN.encode(),
    ):
        raise HTTPException(status_code=403, detail="Forbidden")


# ---------------------------------------------------------------------------
# Subprocess helper — list-form only, never shell=True
# ---------------------------------------------------------------------------


def _run(cmd: list[str], timeout: int = 5) -> tuple[int, str, str]:
    """
    Execute a command as a list (no shell interpolation possible).
    All callers must pass fully-hardcoded argument lists — no user input
    should ever be interpolated into cmd.
    """
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,  # explicit — never allow shell expansion
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except FileNotFoundError:
        return -1, "", "command not found"


def _sanitise(text: str) -> str:
    """Strip ANSI escape sequences and non-printable control characters."""
    text = _ANSI_ESCAPE.sub("", text)
    text = _CONTROL_CHARS.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Data collectors — all subprocess args are hardcoded, never user-supplied
# ---------------------------------------------------------------------------


def _systemd_service_info(name: str) -> dict:
    # name is always from the SERVICES whitelist — validated before this call
    rc, out, _ = _run(
        [
            "systemctl",
            "show",
            name,
            "--property=ActiveState,SubState,MainPID,MemoryCurrent,ExecMainStartTimestamp",
        ]
    )
    props: dict = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            props[k] = v

    active = props.get("ActiveState", "unknown")
    mem_mb: float | None = None
    try:
        mem_mb = round(int(props.get("MemoryCurrent", "")) / 1024 / 1024, 1)
    except (ValueError, TypeError):
        pass

    pid = props.get("MainPID", "0")
    return {
        "name": name,
        "active": active,
        "sub": props.get("SubState", "unknown"),
        "pid": pid if pid != "0" else None,
        "memory_mb": mem_mb,
        "started": props.get("ExecMainStartTimestamp") or None,
        "ok": active == "active",
    }


def _systemd_timer_info(name: str) -> dict:
    rc, out, _ = _run(
        [
            "systemctl",
            "show",
            name,
            "--property=ActiveState,NextElapseUSecRealtime,LastTriggerUSec",
        ]
    )
    props: dict = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            props[k] = v

    return {
        "name": name,
        "active": props.get("ActiveState", "unknown"),
        "next_elapse": props.get("NextElapseUSecRealtime") or None,
        "last_trigger": props.get("LastTriggerUSec") or None,
        "ok": props.get("ActiveState") == "active",
    }


def _docker_container_info(name: str) -> dict:
    # name is always from the CONTAINERS whitelist — validated before this call
    # Go template is hardcoded — no user input reaches docker inspect
    rc, out, _ = _run(
        [
            "docker",
            "inspect",
            "-f",
            "{{.State.Status}}|{{.State.StartedAt}}|{{.State.Health.Status}}",
            name,
        ]
    )
    if rc != 0 or not out:
        return {"name": name, "status": "not found", "ok": False}
    parts = out.split("|")
    status = parts[0] if parts else "unknown"
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
        "timestamp": datetime.now(UTC).isoformat(),
        "hostname": "",
        "uptime": "",
        "load": [],
        "memory": {},
        "disk": {},
        "cpu_temp_c": None,
        "gpu": None,
    }

    rc, out, _ = _run(["hostname"])
    metrics["hostname"] = _sanitise(out)

    try:
        with open("/proc/uptime") as f:
            secs = float(f.read().split()[0])
            h, rem = divmod(int(secs), 3600)
            metrics["uptime"] = f"{h}h {rem // 60}m"
    except Exception:
        pass

    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
            metrics["load"] = [float(parts[0]), float(parts[1]), float(parts[2])]
    except Exception:
        pass

    try:
        rc, out, _ = _run(["free", "-m"])
        for line in out.splitlines():
            if line.startswith("Mem:"):
                parts = line.split()
                total, used = int(parts[1]), int(parts[2])
                metrics["memory"] = {
                    "total_mb": total,
                    "used_mb": used,
                    "free_mb": int(parts[3]),
                    "pct": round(used * 100 / total, 1) if total else 0,
                }
    except Exception:
        pass

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

    try:
        temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if temp_path.exists():
            metrics["cpu_temp_c"] = round(int(temp_path.read_text().strip()) / 1000, 1)
    except Exception:
        pass

    try:
        rc, out, _ = _run(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        if rc == 0 and out:
            parts = [p.strip() for p in out.split(",")]
            metrics["gpu"] = {
                "name": _sanitise(parts[0]),
                "temp_c": int(parts[1]),
                "util_pct": int(parts[2]),
                "vram_used_mb": int(parts[3]),
                "vram_total_mb": int(parts[4]),
            }
    except Exception:
        pass

    return metrics


def _qdrant_info() -> dict:
    """Query Qdrant HTTP API directly and return collection stats per desk."""
    import urllib.error
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY

    def _get(path: str) -> dict | None:
        req = urllib.request.Request(f"{QDRANT_URL}{path}", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return None

    top = _get("/collections")
    if not top or top.get("status") != "ok":
        return {"ok": False, "error": "Qdrant unreachable", "collections": []}

    existing = {c["name"] for c in top.get("result", {}).get("collections", [])}
    collections = []
    total_points = 0

    for name in sorted(existing):
        info = _get(f"/collections/{name}")
        result = (info or {}).get("result", {})
        points = result.get("points_count", 0) or 0
        vectors = result.get("vectors_count", 0) or 0
        segments = result.get("segments_count", 0) or 0
        total_points += points
        collections.append(
            {
                "name": name,
                "is_desk": name in QDRANT_DESK_COLLECTIONS,
                "points_count": points,
                "vectors_count": vectors,
                "segments_count": segments,
                "status": result.get("status", "unknown"),
            }
        )

    return {
        "ok": True,
        "url": QDRANT_URL,
        "total_collections": len(existing),
        "total_points": total_points,
        "collections": collections,
    }


def _redis_info() -> dict:
    rc, out, _ = _run(["docker", "exec", "osia-redis", "redis-cli", "ping"])
    if "PONG" not in out:
        return {"ok": False, "queue_depth": None}

    rc2, depth, _ = _run(["docker", "exec", "osia-redis", "redis-cli", "LLEN", "osia:task_queue"])
    queue_depth = int(depth) if depth.isdigit() else 0

    # Peek at the next task but extract only safe metadata fields —
    # never return raw payload content (may contain Signal numbers, keys, etc.)
    task_meta: dict | None = None
    if queue_depth > 0:
        rc3, peek, _ = _run(["docker", "exec", "osia-redis", "redis-cli", "LINDEX", "osia:task_queue", "0"])
        if peek:
            try:
                import json

                task = json.loads(peek)
                # Only surface non-sensitive routing fields
                task_meta = {k: task[k] for k in ("type", "source", "desk", "timestamp") if k in task}
            except Exception:
                task_meta = {"type": "unparseable"}

    return {
        "ok": True,
        "queue_depth": queue_depth,
        "next_task_meta": task_meta,
    }


def _get_logs(service: str, lines: int) -> list[str]:
    # service is pre-validated against SERVICES whitelist by the route handler
    # lines is a clamped integer — safe to pass as string arg
    rc, out, err = _run(
        ["journalctl", "-u", service, "-n", str(lines), "--no-pager", "--output=short-iso"],
        timeout=10,
    )
    if rc != 0:
        return [f"ERROR: journalctl returned {rc}"]
    # Sanitise every line — strip ANSI and control chars from log output
    return [_sanitise(line) for line in out.splitlines()]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/status")
@limiter.limit("30/minute")
async def get_status(request: Request, _=Depends(_check_auth)):
    """Full system status snapshot."""
    system, services, timers, containers, redis_info, qdrant_info = await asyncio.gather(
        asyncio.to_thread(_system_metrics),
        asyncio.gather(*[asyncio.to_thread(_systemd_service_info, s) for s in SERVICES]),
        asyncio.gather(*[asyncio.to_thread(_systemd_timer_info, t) for t in TIMERS]),
        asyncio.gather(*[asyncio.to_thread(_docker_container_info, c) for c in CONTAINERS]),
        asyncio.to_thread(_redis_info),
        asyncio.to_thread(_qdrant_info),
    )
    return {
        "system": system,
        "services": list(services),
        "timers": list(timers),
        "containers": list(containers),
        "redis": redis_info,
        "qdrant": qdrant_info,
    }


@app.get("/status/services")
@limiter.limit("30/minute")
async def get_services(request: Request, _=Depends(_check_auth)):
    results = await asyncio.gather(*[asyncio.to_thread(_systemd_service_info, s) for s in SERVICES])
    return {"services": list(results)}


@app.get("/status/containers")
@limiter.limit("30/minute")
async def get_containers(request: Request, _=Depends(_check_auth)):
    results = await asyncio.gather(*[asyncio.to_thread(_docker_container_info, c) for c in CONTAINERS])
    return {"containers": list(results)}


@app.get("/status/system")
@limiter.limit("30/minute")
async def get_system(request: Request, _=Depends(_check_auth)):
    return await asyncio.to_thread(_system_metrics)


@app.get("/status/qdrant")
@limiter.limit("30/minute")
async def get_qdrant(request: Request, _=Depends(_check_auth)):
    return await asyncio.to_thread(_qdrant_info)


@app.get("/status/redis")
@limiter.limit("30/minute")
async def get_redis(request: Request, _=Depends(_check_auth)):
    return await asyncio.to_thread(_redis_info)


@app.get("/logs/{service}")
@limiter.limit("20/minute")
async def get_service_logs(service: str, request: Request, lines: int = 100, _=Depends(_check_auth)):
    """
    Tail logs for a named systemd service.
    Service name may omit the .service suffix.
    Max 500 lines per request.
    """
    if not service.endswith(".service"):
        service = f"{service}.service"
    # Whitelist check — service name never reaches subprocess unless it's in this list
    if service not in SERVICES:
        raise HTTPException(status_code=404, detail="Unknown service")
    lines = max(1, min(lines, 500))
    log_lines = await asyncio.to_thread(_get_logs, service, lines)
    return {"service": service, "lines": log_lines, "count": len(log_lines)}


@app.get("/health")
@limiter.limit("60/minute")
async def health(request: Request):
    """Minimal liveness probe — requires correct UA sentinel, returns no detail."""
    ua = request.headers.get("user-agent", "")
    if STATUS_API_UA_SENTINEL not in ua:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=STATUS_API_PORT,
        server_header=False,  # suppress "server: uvicorn" fingerprint
        date_header=False,
    )
