"""
OSIA Queue API — authenticated HTTP wrapper around Redis for remote job submission.

Exposes the minimal Redis operations needed by external workers (HuggingFace Spaces
or any remote process) without exposing Redis directly:

  POST /queue/push          — push a job onto a named queue (rpush)
  POST /queue/pop           — blocking pop from a queue (blpop, timeout configurable)
  GET  /queue/length        — queue depth (llen)
  POST /queue/seen/check    — check if a URL/ID has been seen (sismember)
  POST /queue/seen/add      — mark a URL/ID as seen (sadd)
  GET  /health              — unauthenticated liveness probe

Authentication:
  - Bearer token (QUEUE_API_TOKEN env var), constant-time comparison
  - User-Agent sentinel (QUEUE_API_UA_SENTINEL) — wrong UA returns 404

Allowed queues are whitelisted — callers cannot push to arbitrary Redis keys.

Token management:
  uv run python scripts/manage_status_token.py --service queue
"""

import hmac
import json
import logging
import os
from typing import Any

import redis.asyncio as aioredis
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

logger = logging.getLogger("osia.queue_api")

QUEUE_API_TOKEN = os.getenv("QUEUE_API_TOKEN", "")
QUEUE_API_UA_SENTINEL = os.getenv("QUEUE_API_UA_SENTINEL", "osia-worker/1")
QUEUE_API_PORT = int(os.getenv("QUEUE_API_PORT", "8098"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Whitelist of queue keys remote workers are allowed to touch.
# Nothing outside this list can be pushed to or popped from.
ALLOWED_QUEUES: set[str] = {
    "osia:task_queue",
    "osia:research_queue",
}

# Whitelist of seen-set keys (for deduplication tracking)
ALLOWED_SEEN_SETS: set[str] = {
    "osia:rss:seen_links",
    "osia:research:seen_topics",
}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
security = HTTPBearer()

_redis: aioredis.Redis | None = None


@app.on_event("startup")
async def _startup():
    global _redis
    _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Queue API connected to Redis at %s", REDIS_URL)


@app.on_event("shutdown")
async def _shutdown():
    if _redis:
        await _redis.aclose()


def _get_redis() -> aioredis.Redis:
    if _redis is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    return _redis


# ---------------------------------------------------------------------------
# Middleware — strip server fingerprint
# ---------------------------------------------------------------------------


@app.middleware("http")
async def _remove_server_header(request: Request, call_next):
    response = await call_next(request)
    response.headers.__delitem__("server") if "server" in response.headers else None
    response.headers.__delitem__("x-powered-by") if "x-powered-by" in response.headers else None
    return response


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _check_auth(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not QUEUE_API_TOKEN:
        raise HTTPException(status_code=503, detail="Not configured")
    ua = request.headers.get("user-agent", "")
    if QUEUE_API_UA_SENTINEL not in ua:
        raise HTTPException(status_code=404, detail="Not found")
    if not hmac.compare_digest(
        credentials.credentials.encode(),
        QUEUE_API_TOKEN.encode(),
    ):
        raise HTTPException(status_code=403, detail="Forbidden")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class PushRequest(BaseModel):
    queue: str
    payload: dict[str, Any]


class PopRequest(BaseModel):
    queue: str
    timeout: int = 10  # seconds to block; 0 = non-blocking


class SeenCheckRequest(BaseModel):
    key: str
    member: str


class SeenAddRequest(BaseModel):
    key: str
    members: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
@limiter.limit("60/minute")
async def health(request: Request):
    ua = request.headers.get("user-agent", "")
    if QUEUE_API_UA_SENTINEL not in ua:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


@app.post("/queue/push")
@limiter.limit("120/minute")
async def push(request: Request, body: PushRequest, _=Depends(_check_auth)):
    """Push a job payload onto a queue (rpush)."""
    if body.queue not in ALLOWED_QUEUES:
        raise HTTPException(status_code=400, detail="Queue not permitted")
    r = _get_redis()
    length = await r.rpush(body.queue, json.dumps(body.payload))
    logger.info("Pushed job to %s (depth now %d)", body.queue, length)
    return {"ok": True, "queue": body.queue, "depth": length}


@app.post("/queue/pop")
@limiter.limit("120/minute")
async def pop(request: Request, body: PopRequest, _=Depends(_check_auth)):
    """
    Pop the next job from a queue (blpop with timeout).
    Returns null payload if the queue is empty and timeout expires.
    """
    if body.queue not in ALLOWED_QUEUES:
        raise HTTPException(status_code=400, detail="Queue not permitted")
    timeout = max(0, min(body.timeout, 30))  # cap at 30s
    r = _get_redis()
    result = await r.blpop(body.queue, timeout=timeout)
    if result is None:
        return {"ok": True, "queue": body.queue, "payload": None}
    _, raw = result
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"raw": raw}
    return {"ok": True, "queue": body.queue, "payload": payload}


@app.get("/queue/length")
@limiter.limit("60/minute")
async def queue_length(request: Request, queue: str, _=Depends(_check_auth)):
    """Return the current depth of a queue (llen)."""
    if queue not in ALLOWED_QUEUES:
        raise HTTPException(status_code=400, detail="Queue not permitted")
    r = _get_redis()
    depth = await r.llen(queue)
    return {"ok": True, "queue": queue, "depth": depth}


@app.post("/queue/seen/check")
@limiter.limit("120/minute")
async def seen_check(request: Request, body: SeenCheckRequest, _=Depends(_check_auth)):
    """Check if a member exists in a seen-set (sismember)."""
    if body.key not in ALLOWED_SEEN_SETS:
        raise HTTPException(status_code=400, detail="Set not permitted")
    r = _get_redis()
    seen = await r.sismember(body.key, body.member)
    return {"ok": True, "key": body.key, "member": body.member, "seen": bool(seen)}


@app.post("/queue/seen/add")
@limiter.limit("120/minute")
async def seen_add(request: Request, body: SeenAddRequest, _=Depends(_check_auth)):
    """Add one or more members to a seen-set (sadd)."""
    if body.key not in ALLOWED_SEEN_SETS:
        raise HTTPException(status_code=400, detail="Set not permitted")
    if not body.members:
        raise HTTPException(status_code=400, detail="No members provided")
    r = _get_redis()
    added = await r.sadd(body.key, *body.members)
    return {"ok": True, "key": body.key, "added": added}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=QUEUE_API_PORT,
        server_header=False,
        date_header=False,
    )
