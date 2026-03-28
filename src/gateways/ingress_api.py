"""
OSIA Ingress API — authenticated HTTP interface for submitting intelligence tasks.

Provides a high-level semantic interface for triggering intelligence gathering
and reporting exercises from internal tooling or external callers.

  POST /ingest          — submit an intelligence query → osia:task_queue
  POST /research        — submit a deep-research topic → osia:research_queue
  GET  /queue/status    — queue depths for both queues
  GET  /health          — unauthenticated liveness probe (UA-gated)

Authentication:
  - Bearer token (INGRESS_API_TOKEN env var), constant-time comparison
  - User-Agent sentinel (INGRESS_API_UA_SENTINEL) — wrong UA returns 404

Token management:
  uv run python scripts/manage_ingress_token.py --rotate
"""

import hashlib
import hmac
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal

import redis.asyncio as aioredis
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

logger = logging.getLogger("osia.ingress_api")

INGRESS_API_TOKEN = os.getenv("INGRESS_API_TOKEN", "")
INGRESS_API_UA_SENTINEL = os.getenv("INGRESS_API_UA_SENTINEL", "osia-ingress/1")
INGRESS_API_PORT = int(os.getenv("INGRESS_API_PORT", "8097"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

TASK_QUEUE = "osia:task_queue"
RESEARCH_QUEUE = "osia:research_queue"

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

_redis: aioredis.Redis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis
    _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Ingress API connected to Redis at %s", REDIS_URL)
    yield
    if _redis:
        await _redis.aclose()


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
security = HTTPBearer()


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
    if "server" in response.headers:
        del response.headers["server"]
    if "x-powered-by" in response.headers:
        del response.headers["x-powered-by"]
    return response


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _check_auth(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not INGRESS_API_TOKEN:
        raise HTTPException(status_code=503, detail="Not configured")
    ua = request.headers.get("user-agent", "")
    if INGRESS_API_UA_SENTINEL not in ua:
        raise HTTPException(status_code=404, detail="Not found")
    if not hmac.compare_digest(
        credentials.credentials.encode(),
        INGRESS_API_TOKEN.encode(),
    ):
        raise HTTPException(status_code=403, detail="Forbidden")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="The intelligence query or URL to analyse")
    label: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_\-\.]+$",
        description="Optional caller label (alphanumeric, hyphens, underscores, dots). Appears as the task source.",
    )
    priority: Literal["normal", "high"] = Field(
        default="normal",
        description="Task priority hint ('normal' or 'high'). High-priority tasks are prepended to the queue.",
    )
    desk: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9-]*[a-z0-9]$",
        description="Optional desk slug to bypass AI routing and send directly to a named desk.",
    )

    model_config = {"str_strip_whitespace": True}


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=2000, description="Research topic to enqueue")
    label: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_\-\.]+$",
        description="Optional caller label. Used as dedup key prefix.",
    )
    desk: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9-]*[a-z0-9]$",
        description="Optional desk slug. Determines model routing in the research worker (e.g. cyber gets mistral-31-24b). Defaults to AI-selected routing.",
    )

    model_config = {"str_strip_whitespace": True}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
@limiter.limit("60/minute")
async def health(request: Request):
    """Liveness probe — UA-gated, returns no system detail."""
    ua = request.headers.get("user-agent", "")
    if INGRESS_API_UA_SENTINEL not in ua:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


@app.post("/ingest")
@limiter.limit("30/minute")
async def ingest(request: Request, body: IngestRequest, _=Depends(_check_auth)):
    """
    Submit an intelligence query for immediate processing.

    The query is pushed onto the orchestrator's task queue as a source-tagged
    task. High-priority tasks are prepended (lpush) so they run ahead of the
    normal backlog; normal tasks are appended (rpush).

    Returns the generated task_id and resulting queue depth.
    """
    r = _get_redis()

    label = body.label or "external"
    source = f"api:{label}"
    task_id = str(uuid.uuid4())

    task = {
        "source": source,
        "query": body.query,
        "task_id": task_id,
        "timestamp": datetime.now(UTC).isoformat(),
        **({"desk": body.desk} if body.desk else {}),
    }

    payload = json.dumps(task)
    if body.priority == "high":
        depth = await r.lpush(TASK_QUEUE, payload)
    else:
        depth = await r.rpush(TASK_QUEUE, payload)

    # Log only internally-generated values — task_id and depth are not user-controlled
    logger.info("Ingress task queued [%s] depth=%d", task_id, depth)
    return {"ok": True, "task_id": task_id, "queue": TASK_QUEUE, "queue_depth": depth}


@app.post("/research")
@limiter.limit("20/minute")
async def research(request: Request, body: ResearchRequest, _=Depends(_check_auth)):
    """
    Submit a topic for deep background research.

    The topic is pushed onto the research worker queue. Deduplication is
    enforced via a TTL-keyed Redis set — duplicate topics within the cooldown
    window (default 24 h) are silently skipped.

    Returns whether the topic was queued or skipped as a duplicate.
    """
    r = _get_redis()

    label = body.label or "external"
    dedup_key = hashlib.md5(body.topic.lower().strip().encode(), usedforsecurity=False).hexdigest()
    seen_key = f"osia:research:seen:{dedup_key}"

    already_seen = await r.exists(seen_key)
    if already_seen:
        depth = await r.llen(RESEARCH_QUEUE)
        logger.debug("Research dedup skip: topic already seen (key=%s)", seen_key)
        return {"ok": True, "queued": False, "reason": "duplicate", "queue_depth": depth}

    task = {
        "topic": body.topic,
        "source": f"api:{label}",
        "timestamp": datetime.now(UTC).isoformat(),
        **({"desk": body.desk} if body.desk else {}),
    }
    depth = await r.rpush(RESEARCH_QUEUE, json.dumps(task))

    # Mirror the research worker's dedup TTL (default 24 h)
    cooldown_hours = int(os.getenv("RESEARCH_COOLDOWN_HOURS", "24"))
    await r.setex(seen_key, cooldown_hours * 3600, "1")

    # Log only internally-generated depth — label is user-controlled
    logger.info("Research topic queued depth=%d", depth)
    return {"ok": True, "queued": True, "queue": RESEARCH_QUEUE, "queue_depth": depth}


@app.get("/queue/status")
@limiter.limit("60/minute")
async def queue_status(request: Request, _=Depends(_check_auth)):
    """Return current depth of both the task and research queues."""
    r = _get_redis()
    task_depth, research_depth = await r.llen(TASK_QUEUE), await r.llen(RESEARCH_QUEUE)
    return {
        "ok": True,
        "queues": {
            TASK_QUEUE: task_depth,
            RESEARCH_QUEUE: research_depth,
        },
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=INGRESS_API_PORT,
        server_header=False,
        date_header=False,
    )
