"""
OSIA Research Worker — oneshot batch processor, runs locally via systemd timer.

Drains osia:research_queue, runs a multi-turn tool-calling research loop via
Venice AI (uncensored/permissive model routing per desk), then chunks and
embeds results into Qdrant for retrieval-augmented generation at report time.

Desk → model routing:
  HUMINT / Cultural / Geopolitical / InfoWar  → venice-uncensored          (no guardrails, ReAct tool use)
  Cyber / Finance                             → mistral-small-3-2-24b-instruct (Venice, native FC)
  Science / Environment / default             → mistral-small-3-2-24b-instruct (cheap, native FC)

Fallback chain: Venice → OpenRouter → Gemini

Environment variables:
  VENICE_API_KEY              — Venice API key (primary)
  OPENROUTER_API_KEY          — OpenRouter API key (fallback)
  REDIS_URL                   — Redis connection URL (default: redis://localhost:6379/0)
  QDRANT_URL                  — Qdrant HTTP endpoint (default: https://qdrant.osia.dev)
  QDRANT_API_KEY              — Qdrant API key
  HF_TOKEN                    — HuggingFace token (for embeddings)
  TAVILY_API_KEY              — Tavily web search API key
  RESEARCH_BATCH_THRESHOLD    — Min queue depth before processing (default: 3)
  RESEARCH_MAX_JOBS_PER_RUN   — Max jobs processed per run (default: 25)
  RESEARCH_COOLDOWN_HOURS     — Hours before a topic can be re-researched (default: 72)
  VENICE_MODEL_UNCENSORED     — Override uncensored model slug (default: venice-uncensored)
  VENICE_MODEL_CYBER          — Override cyber desk model slug (default: mistral-small-3-2-24b-instruct)
  VENICE_MODEL_DEFAULT        — Override default model slug (default: mistral-small-3-2-24b-instruct)
  GEMINI_API_KEY              — Fallback if neither Venice nor OpenRouter key is set
  GEMINI_MODEL_ID             — Gemini model ID (default: gemini-2.5-flash)

Run:
  uv run python -m src.workers.research_worker
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from dotenv import load_dotenv

from src.intelligence.wiki_client import WikiClient, build_entity_page, desk_wiki_section, entity_wiki_path

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("osia.research_worker")

# DDGS logs INFO for every backend engine failure it handles internally (Bing
# DecodeErrors, Yahoo RequestErrors, etc.).  These are expected transient noise
# — the library retries other engines automatically, so suppress below WARNING.
for _ddgs_logger in ("ddgs", "ddgs.ddgs", "ddgs.engines", "ddgs.engines.yahoo_news"):
    logging.getLogger(_ddgs_logger).setLevel(logging.ERROR)


def _http_error_detail(exc: httpx.HTTPStatusError) -> str:
    """Extract a human-readable error message from an API error response."""
    try:
        body = exc.response.json()
        err = body.get("error")
        if isinstance(err, dict):
            return err.get("message") or str(err)
        if err:
            return str(err)
        if body.get("detail"):
            return str(body["detail"])
        if body.get("message"):
            return str(body["message"])
    except Exception:
        pass  # response body is not JSON — fall through to raw text
    return exc.response.text[:300]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VENICE_API_KEY = os.getenv("VENICE_API_KEY", "")
VENICE_BASE_URL = "https://api.venice.ai/api/v1"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

S2_API_KEY = os.getenv("S2_API_KEY", "")  # Semantic Scholar — free key lifts limit to 10 req/s
CENSYS_API_ID = os.getenv("CENSYS_API_ID", "")
CENSYS_API_SECRET = os.getenv("CENSYS_API_SECRET", "")
CRIMINALIP_API_KEY = os.getenv("CRIMINALIP_API_KEY", "")
OTX_API_KEY = os.getenv("OTX_API_KEY", "")
ALEPH_API_KEY = os.getenv("ALEPH_API_KEY", "")  # Optional — public datasets accessible without key

# Camoufox (real Firefox browser) — used when robots.txt disallows crawling or simpler
# fetchers hit bot-detection walls.  Set CAMOUFOX_HEADLESS=false on desktop Linux.
CAMOUFOX_HEADLESS = os.getenv("CAMOUFOX_HEADLESS", "true").lower() not in ("0", "false", "no")

# Per-process semaphore — one Camoufox instance at a time to cap memory use.
_camoufox_sem = asyncio.Semaphore(1)

# Domain-level robots.txt cache: domain → (crawl_allowed, expires_at)
_robots_cache: dict[str, tuple[bool, float]] = {}
_ROBOTS_TTL = 3600.0  # re-check each domain at most once per hour

WIKIPEDIA_USER_AGENT = os.getenv(
    "WIKIPEDIA_USER_AGENT",
    "OSIA-Intelligence-Framework/1.0 (https://osia.dev; research@osia.dev) python-httpx",
)

BATCH_THRESHOLD = int(os.getenv("RESEARCH_BATCH_THRESHOLD", "3"))
RESEARCH_COOLDOWN_SECONDS = int(os.getenv("RESEARCH_COOLDOWN_HOURS", "72")) * 3600
# Maximum jobs processed in a single run.  Keeps each systemd oneshot bounded so
# the timer can re-fire before the queue is fully drained.  The remaining items
# stay in Redis and are picked up on the next run.
MAX_JOBS_PER_RUN = int(os.getenv("RESEARCH_MAX_JOBS_PER_RUN", "25"))

# Tavily is a paid API with a monthly quota.  We track usage in Redis and
# refuse Tavily calls once the budget is exhausted, falling back to DuckDuckGo.
TAVILY_MONTHLY_BUDGET = int(os.getenv("TAVILY_MONTHLY_BUDGET", "500"))

# OpenRouter model used when falling back from Venice at runtime.
# Must be an OpenRouter-compatible model ID — Venice slugs (e.g. venice-uncensored)
# are not valid on OpenRouter.
OPENROUTER_RESEARCH_MODEL = os.getenv("RESEARCH_OPENROUTER_MODEL", "google/gemini-2.5-flash")

# Venice model routing per desk
VENICE_MODEL_UNCENSORED = os.getenv("VENICE_MODEL_UNCENSORED", "venice-uncensored")
VENICE_MODEL_CYBER = os.getenv("VENICE_MODEL_CYBER", "mistral-small-3-2-24b-instruct")
VENICE_MODEL_DEFAULT = os.getenv("VENICE_MODEL_DEFAULT", "mistral-small-3-2-24b-instruct")

# Models that reject tool_choice/tools in the API payload — use ReAct prompt only
REACT_ONLY_MODELS = {VENICE_MODEL_UNCENSORED}

# Desks that require uncensored reasoning (no guardrails)
UNCENSORED_DESKS = {
    "human-intelligence-and-profiling-desk",
    "cultural-and-theological-intelligence-desk",
    "geopolitical-and-security-desk",
    "information-warfare-desk",
}
CYBER_DESKS = {
    "cyber-intelligence-and-warfare-desk",
}
FINANCE_DESKS = {
    "finance-and-economics-directorate",
}

RESEARCH_COLLECTION = "osia_research_cache"
EMBEDDING_DIM = 384
CHUNK_SIZE = 400  # words
MAX_ROUNDS = 6

# ---------------------------------------------------------------------------
# Desk-aware KB collection resolution
# ---------------------------------------------------------------------------
# Load boost_collections from each desk YAML so the research worker automatically
# picks up any future changes without needing a code update.


def _load_desk_boost_collections() -> dict[str, list[str]]:
    """Parse every config/desks/*.yaml and return desk_slug → boost_collections."""
    import pathlib

    import yaml

    result: dict[str, list[str]] = {}
    desk_dir = pathlib.Path(__file__).parent.parent.parent / "config" / "desks"
    for yaml_path in sorted(desk_dir.glob("*.yaml")):
        try:
            with yaml_path.open() as fh:
                cfg = yaml.safe_load(fh) or {}
            slug = cfg.get("qdrant", {}).get("collection", "")
            boost = cfg.get("qdrant", {}).get("boost_collections") or []
            if slug:
                result[slug] = list(boost)
        except Exception as exc:
            logger.warning("Could not load desk YAML %s: %s", yaml_path.name, exc)
    return result


_DESK_BOOST: dict[str, list[str]] = _load_desk_boost_collections()


def _collections_for_desk(desk: str) -> list[str]:
    """Return the ordered, deduplicated list of Qdrant collections for a desk.

    Always includes the full DESK_COLLECTIONS baseline (all desk primaries + broad
    KBs). Appends desk-specific boost collections on top, so the research worker
    searches exactly the same specialised sources as the orchestrator's RAG layer.
    """
    from src.intelligence.qdrant_store import DESK_COLLECTIONS

    extras = _DESK_BOOST.get(desk, [])
    seen: set[str] = set()
    result: list[str] = []
    for col in DESK_COLLECTIONS + extras:
        if col not in seen:
            seen.add(col)
            result.append(col)
    return result


def _make_kb_tool(desk: str):
    """Return a desk-aware search_intel_kb tool function.

    The returned coroutine searches DESK_COLLECTIONS plus any boost collections
    declared in the desk's YAML — giving the research worker the same Qdrant
    coverage as the orchestrator's RAG context block.
    """
    collections = _collections_for_desk(desk)

    async def _tool(query: str, _http) -> str:
        try:
            from src.intelligence.qdrant_store import QdrantStore

            async with QdrantStore() as store:
                results = await store.cross_desk_search(query, top_k=5, collections=collections)
            results = [r for r in results if r.score >= 0.45]
            if not results:
                return "No relevant intel found in the knowledge base for this query."
            parts = []
            for r in results:
                source = r.metadata.get("source", r.collection)
                date = r.metadata.get("collected_at", r.metadata.get("date", ""))
                header = f"[KB: {r.collection} | score={r.score:.2f} | source={source}"
                if date:
                    header += f" | {date[:10]}"
                header += "]"
                parts.append(f"{header}\n{r.text[:600]}")
            return "\n\n---\n\n".join(parts)
        except Exception as e:
            return f"Intel KB search error: {e}"

    return _tool


def get_tool_registry(desk: str) -> dict:
    """Return a tool registry with a desk-specific search_intel_kb.

    All other tools are shared; only search_intel_kb is overridden to include
    the desk's boost collections in its Qdrant fan-out.
    """
    return {**TOOL_REGISTRY, "search_intel_kb": _make_kb_tool(desk)}


def _model_for_desk(desk: str) -> str:
    if desk in UNCENSORED_DESKS:
        return VENICE_MODEL_UNCENSORED
    if desk in CYBER_DESKS:
        return VENICE_MODEL_CYBER
    if desk in FINANCE_DESKS:
        return VENICE_MODEL_CYBER  # same tier as Cyber
    return VENICE_MODEL_DEFAULT


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ResearchJob:
    job_id: str
    topic: str
    desk: str
    priority: str = "normal"  # critical | high | normal | low
    directives_lens: bool = True
    triggered_by: str = ""
    source: str = ""  # set by ingress API (e.g. "api:external"); triggered_by used elsewhere
    entity_type: str = ""  # Person | Organisation | Location | Event | Technology | Concept

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchJob":
        return cls(
            job_id=d.get("job_id", str(uuid.uuid4())),
            topic=d.get("topic", ""),
            desk=d.get("desk", "collection-directorate"),
            priority=d.get("priority", "normal"),
            directives_lens=d.get("directives_lens", True),
            triggered_by=d.get("triggered_by", ""),
            source=d.get("source", ""),
            entity_type=d.get("entity_type", ""),
        )


# ---------------------------------------------------------------------------
# Priority queue — sorted set scoring
# ---------------------------------------------------------------------------

# Lower score = processed first.  Tier × 10¹² keeps tiers well-separated;
# Unix timestamp within each tier preserves FIFO order.
PRIORITY_SCORES: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "normal": 2,
    "low": 3,
}


def _priority_score(priority: str) -> int:
    tier = PRIORITY_SCORES.get(priority, 2)
    return tier * 1_000_000_000_000 + int(time.time())


# ---------------------------------------------------------------------------
# Redis client (direct — no HTTP queue API needed when running locally)
# ---------------------------------------------------------------------------

RESEARCH_QUEUE = "osia:research_queue"
PROCESSING_LIST = "osia:research:processing"

# Atomically pops the lowest-score member from a sorted set (KEYS[1]) and
# appends it to a list (KEYS[2]).  Returns the member string, or nil if empty.
_LUA_ZPOP_MOVE = """
local result = redis.call('ZPOPMIN', KEYS[1], 1)
if #result == 0 then return nil end
local member = result[1]
redis.call('RPUSH', KEYS[2], member)
return member
"""


class RedisQueue:
    def __init__(self):
        import redis

        self._r = redis.from_url(REDIS_URL, decode_responses=True)
        self._zpop_move = self._r.register_script(_LUA_ZPOP_MOVE)
        self._migrate_list_to_zset()
        self._recover_stranded_jobs()

    def _migrate_list_to_zset(self) -> None:
        """One-time migration: convert legacy list to sorted set on first run."""
        key_type = self._r.type(RESEARCH_QUEUE)
        if key_type == "none" or key_type == "zset":
            return
        if key_type != "list":
            logger.warning("Unexpected Redis type for %s: %s — leaving as-is", RESEARCH_QUEUE, key_type)
            return

        items = self._r.lrange(RESEARCH_QUEUE, 0, -1)
        logger.info("Migrating %d list items in %s to priority sorted set", len(items), RESEARCH_QUEUE)
        pipe = self._r.pipeline()
        pipe.delete(RESEARCH_QUEUE)
        base_ts = int(time.time())
        for i, item in enumerate(items):
            score = PRIORITY_SCORES["normal"] * 1_000_000_000_000 + base_ts + i
            pipe.zadd(RESEARCH_QUEUE, {item: score})
        pipe.execute()
        logger.info("Migration complete — %d items re-enqueued as normal priority", len(items))

    def _recover_stranded_jobs(self) -> None:
        """Re-queue any jobs left in the processing list from a previous crashed run.

        LPOP is atomic so concurrent workers draining the list simultaneously is
        safe — each item is popped exactly once regardless of how many workers call
        this at startup.
        """
        recovered = 0
        score_base = PRIORITY_SCORES["normal"] * 1_000_000_000_000 + int(time.time())
        while True:
            raw = self._r.lpop(PROCESSING_LIST)
            if raw is None:
                break
            self._r.zadd(RESEARCH_QUEUE, {raw: score_base + recovered})
            try:
                topic = json.loads(raw).get("topic", "?")[:60]
            except Exception:
                topic = "?"
            logger.info("Recovered stranded job: %s", topic)
            recovered += 1
        if recovered:
            logger.info("Re-queued %d stranded job(s) into research queue", recovered)

    def depth(self) -> int:
        return self._r.zcard(RESEARCH_QUEUE)

    def processing_depth(self) -> int:
        return self._r.llen(PROCESSING_LIST)

    def pop(self) -> tuple[dict, str] | None:
        """Atomically move the highest-priority job into the processing list.

        Uses a Lua script to ZPOPMIN from the sorted set and RPUSH into the
        processing list in a single round-trip, so no job is ever lost or
        visible to two workers simultaneously.
        Returns (parsed_payload, raw_json) so the caller can pass raw to complete().
        """
        raw = self._zpop_move(keys=[RESEARCH_QUEUE, PROCESSING_LIST])
        if raw is None:
            return None
        try:
            return json.loads(raw), raw
        except json.JSONDecodeError:
            self._r.lrem(PROCESSING_LIST, 1, raw)
            return None

    def push(self, payload: dict, priority: str = "normal") -> int:
        """Enqueue a job with the given priority. Returns current queue depth."""
        score = _priority_score(priority)
        self._r.zadd(RESEARCH_QUEUE, {json.dumps(payload): score})
        return self._r.zcard(RESEARCH_QUEUE)

    def complete(self, raw: str) -> None:
        """Remove a job from the processing list after success or failure."""
        self._r.lrem(PROCESSING_LIST, 1, raw)

    def release_inflight(self, topic: str) -> None:
        """Remove a topic from the in-flight guard without marking it as seen.

        Called on failure paths so the entity extractor can re-queue the topic
        on a future poll. mark_seen() handles the success path via its own srem.
        """
        normalised = topic.lower().strip()
        if normalised:
            self._r.srem("osia:research:queued_topics", normalised)

    def is_seen(self, key: str) -> bool:
        return bool(self._r.exists(f"osia:research:seen:{key}"))

    def mark_seen(self, key: str, topic: str = "") -> None:
        normalised = topic.lower().strip()
        self._r.set(f"osia:research:seen:{key}", "1", ex=RESEARCH_COOLDOWN_SECONDS)
        if normalised:
            # Keep seen_topics in sync so entity_extractor dedup survives Redis restarts.
            self._r.sadd("osia:research:seen_topics", normalised)
            # Remove from the in-flight guard set so the topic can be re-queued
            # by the entity extractor once the cooldown expires.
            self._r.srem("osia:research:queued_topics", normalised)


# ---------------------------------------------------------------------------
# Qdrant client
# ---------------------------------------------------------------------------


class QdrantClient:
    def __init__(self, http: httpx.AsyncClient):
        self._http = http
        self._headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}

    async def ensure_collection(self, collection: str = RESEARCH_COLLECTION):
        check = await self._http.get(
            f"{QDRANT_URL}/collections/{collection}",
            headers=self._headers,
        )
        if check.status_code == 200:
            return
        resp = await self._http.put(
            f"{QDRANT_URL}/collections/{collection}",
            headers=self._headers,
            json={
                "vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"},
                "optimizers_config": {"indexing_threshold": 1000},
            },
        )
        resp.raise_for_status()
        logger.info("Created Qdrant collection: %s", collection)

    async def upsert_points(self, points: list[dict], collection: str = RESEARCH_COLLECTION):
        resp = await self._http.put(
            f"{QDRANT_URL}/collections/{collection}/points",
            headers=self._headers,
            json={"points": points},
            timeout=30.0,
        )
        resp.raise_for_status()
        logger.info("Upserted %d points into %s", len(points), collection)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


async def embed_texts(texts: list[str], http: httpx.AsyncClient) -> list[list[float]]:
    try:
        resp = await http.post(
            "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Embedding failed: %s — using zero vectors", e)
        return [[0.0] * EMBEDDING_DIM for _ in texts]


# ---------------------------------------------------------------------------
# Research tools
# ---------------------------------------------------------------------------


async def _crawl_allowed(url: str, http: httpx.AsyncClient) -> bool:
    """Return True if robots.txt permits crawling this URL (or is absent/unreachable).

    Results are cached per domain for _ROBOTS_TTL seconds so we never fetch
    robots.txt more than once per hour per domain.
    """
    try:
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        now = time.time()
        if domain in _robots_cache:
            allowed, expires = _robots_cache[domain]
            if now < expires:
                return allowed
        robots_url = f"{domain}/robots.txt"
        try:
            resp = await http.get(robots_url, timeout=5.0, follow_redirects=True)
            if resp.status_code == 200:
                rp = RobotFileParser()
                rp.parse(resp.text.splitlines())
                allowed = rp.can_fetch("*", url)
            else:
                allowed = True
        except Exception:
            allowed = True
        _robots_cache[domain] = (allowed, now + _ROBOTS_TTL)
        if not allowed:
            logger.debug("robots.txt disallows crawling %s — will use Camoufox", domain)
        return allowed
    except Exception:
        return True


async def _fetch_with_camoufox(url: str) -> str | None:
    """Fetch a URL using a real Firefox browser via Camoufox.

    Used when robots.txt disallows automated crawlers or when simpler fetchers
    hit bot-detection walls (403, Cloudflare, JS-heavy pages).
    Semaphore limits to one concurrent Camoufox instance per worker process.
    CAMOUFOX_HEADLESS env var controls display mode (default: true for servers,
    set to false for desktop Linux runs).
    """
    try:
        import trafilatura
        from camoufox.async_api import AsyncCamoufox
    except ImportError:
        logger.debug("Camoufox not available — skipping")
        return None

    async with _camoufox_sem:
        try:
            async with AsyncCamoufox(headless=CAMOUFOX_HEADLESS) as browser:
                page = await browser.new_page()
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=25000)
                    html = await page.content()
                finally:
                    await page.close()

            text = trafilatura.extract(html, include_comments=False, include_tables=False, favor_recall=True)
            if text:
                logger.info("fetch_url (camoufox) extracted %d chars from %s", len(text), url)
                return f"[Source: {url}]\n\n{text[:15000]}"
            return None
        except Exception as e:
            logger.debug("Camoufox fetch failed for %s: %s", url, e)
            return None


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}


async def tool_fetch_url(url: str, http: httpx.AsyncClient) -> str:
    """Fetch and extract full article text from a URL.

    Tier 0: robots.txt check — if crawling is disallowed, go straight to Camoufox
            (sites that restrict bots actively block simpler fetchers).
    Tier 1: curl_cffi with Chrome TLS fingerprint — defeats most bot detection.
    Tier 2: httpx with browser-like headers — lightweight fallback.
    Tier 3: Camoufox (real Firefox) — for JS-heavy pages or 403/bot-wall responses.
    Extraction: trafilatura → BeautifulSoup plain-text fallback.
    """
    import trafilatura
    from bs4 import BeautifulSoup
    from curl_cffi.requests import AsyncSession

    if not url or not url.startswith(("http://", "https://")):
        return "Error: fetch_url requires a full http:// or https:// URL."

    # Tier 0: robots.txt — skip the simpler fetchers for sites that block bots
    if not await _crawl_allowed(url, http):
        result = await _fetch_with_camoufox(url)
        if result:
            return result
        return f"Error: could not retrieve content from {url} (robots.txt disallows crawling)"

    html: str | None = None
    bot_walled = False

    # Tier 1: curl_cffi
    try:
        async with AsyncSession(impersonate="chrome") as session:
            resp = await session.get(url, headers=_BROWSER_HEADERS, timeout=20, allow_redirects=True)
            if resp.status_code == 200:
                html = resp.text
            elif resp.status_code in (403, 429):
                bot_walled = True
    except Exception as e:
        logger.debug("curl_cffi fetch failed for %s: %s", url, e)

    # Tier 2: httpx
    if not html and not bot_walled:
        try:
            resp = await http.get(url, headers=_BROWSER_HEADERS, timeout=20.0)
            if resp.status_code == 200:
                html = resp.text
            elif resp.status_code in (403, 429):
                bot_walled = True
        except Exception as e:
            logger.debug("httpx article fetch failed for %s: %s", url, e)

    # Tier 3: Camoufox — on bot-wall responses or when both simpler tiers failed
    if not html:
        result = await _fetch_with_camoufox(url)
        if result:
            return result
        return f"Error: could not retrieve content from {url}"

    text = trafilatura.extract(html, include_comments=False, include_tables=False, favor_recall=True)

    if not text:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()
        raw = soup.get_text(separator="\n", strip=True)
        lines = [ln for ln in raw.splitlines() if len(ln.strip()) > 40]
        text = "\n".join(lines)[:20000] or None

    if not text:
        return f"Error: page at {url} yielded no extractable text."

    logger.info("fetch_url extracted %d chars from %s", len(text), url)
    return f"[Source: {url}]\n\n{text[:15000]}"


async def tool_search_web(query: str, _http: httpx.AsyncClient) -> str:
    """DuckDuckGo web search — no API key, no quota. Default for all web queries."""
    try:
        from ddgs import DDGS

        results = await asyncio.to_thread(lambda: list(DDGS().text(query, max_results=5)))
        if not results:
            return "No DuckDuckGo results found."
        return "\n\n".join(f"[{r.get('title', '')}]({r.get('href', '')})\n{r.get('body', '')[:500]}" for r in results)
    except Exception as e:
        return f"DuckDuckGo search error: {e}"


async def tool_search_news(query: str, _http: httpx.AsyncClient) -> str:
    """DuckDuckGo news search — returns recent articles with publication dates and sources.

    Prefer this over search_web when recency matters: breaking events, recent statements,
    current conflict updates, newly published reports. Results include the publication
    date so the model can assess freshness.
    """
    try:
        from ddgs import DDGS

        results = await asyncio.to_thread(lambda: list(DDGS().news(query, max_results=6)))
        if not results:
            return "No news results found."
        return "\n\n".join(
            f"[{r.get('title', '')}]({r.get('url', '')})\n"
            f"Source: {r.get('source', '')} | Published: {r.get('date', '')}\n"
            f"{r.get('body', '')[:400]}"
            for r in results
        )
    except Exception as e:
        return f"News search error: {e}"


async def _tavily_within_budget() -> bool:
    """Increment the monthly Tavily usage counter in Redis and return True if within budget.

    Uses an INCR + EXPIRE pattern so the key self-expires after 35 days.
    Fails open (returns True) if Redis is unavailable, so a connectivity blip
    never silently kills Tavily access.
    """
    key = f"osia:tavily:usage:{datetime.now(UTC).strftime('%Y-%m')}"

    def _redis_incr() -> int:
        import redis as _redis

        r = _redis.from_url(REDIS_URL, decode_responses=True)
        count = r.incr(key)
        if count == 1:
            r.expire(key, 35 * 86400)
        return count

    try:
        count = await asyncio.to_thread(_redis_incr)
    except Exception as e:
        logger.debug("Tavily budget Redis check failed (%s) — allowing call", e)
        return True

    if count > TAVILY_MONTHLY_BUDGET:
        logger.warning(
            "Tavily monthly budget exhausted (%d/%d calls) — routing to DuckDuckGo",
            count - 1,
            TAVILY_MONTHLY_BUDGET,
        )
        return False

    remaining = TAVILY_MONTHLY_BUDGET - count
    if remaining < TAVILY_MONTHLY_BUDGET * 0.15:
        logger.warning("Tavily budget low: %d/%d calls used this month", count, TAVILY_MONTHLY_BUDGET)
    return True


async def tool_search_tavily(query: str, http: httpx.AsyncClient) -> str:
    """Tavily premium search — reserved for queries where recency is critical.

    Checks the monthly budget before calling Tavily; falls back to search_web
    (DuckDuckGo) automatically on budget exhaustion, API errors, or quota
    responses (HTTP 432).
    """
    if not TAVILY_API_KEY:
        logger.info("TAVILY_API_KEY not set — using DuckDuckGo for web search")
        return await tool_search_web(query, http)

    if not await _tavily_within_budget():
        return await tool_search_web(query, http)

    try:
        resp = await http.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 5},
            timeout=15.0,
        )
        if resp.status_code == 432:
            logger.warning("Tavily quota exhausted (432) — falling back to DuckDuckGo")
            return await tool_search_web(query, http)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return "\n\n".join(f"[{r.get('title', '')}]({r.get('url', '')})\n{r.get('content', '')[:500]}" for r in results)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 432:
            logger.warning("Tavily quota exhausted — falling back to DuckDuckGo")
            return await tool_search_web(query, http)
        logger.warning("Tavily HTTP error %d — falling back to DuckDuckGo: %s", e.response.status_code, e)
        return await tool_search_web(query, http)
    except Exception as e:
        logger.warning("Tavily error — falling back to DuckDuckGo: %s", e)
        return await tool_search_web(query, http)


async def tool_search_wikipedia(query: str, http: httpx.AsyncClient) -> str:
    headers = {"User-Agent": WIKIPEDIA_USER_AGENT}
    try:
        resp = await http.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query, "srlimit": 3, "format": "json", "utf8": 1},
            headers=headers,
            timeout=10.0,
        )
        resp.raise_for_status()
        hits = resp.json().get("query", {}).get("search", [])
        if not hits:
            return "No Wikipedia results found."
        title = hits[0]["title"]
        ex = await http.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "titles": title,
                "format": "json",
            },
            headers=headers,
            timeout=10.0,
        )
        ex.raise_for_status()
        pages = ex.json().get("query", {}).get("pages", {})
        extract = next(iter(pages.values()), {}).get("extract", "")
        return f"**{title}**\n\n{extract[:2000]}"
    except Exception as e:
        return f"Wikipedia error: {e}"


async def tool_search_arxiv(query: str, http: httpx.AsyncClient) -> str:
    try:
        resp = await http.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "max_results": 3, "sortBy": "relevance"},
            timeout=15.0,
        )
        resp.raise_for_status()
        entries = re.findall(r"<entry>(.*?)</entry>", resp.text, re.DOTALL)
        results = []
        for e in entries[:3]:
            title = re.search(r"<title>(.*?)</title>", e, re.DOTALL)
            summary = re.search(r"<summary>(.*?)</summary>", e, re.DOTALL)
            link = re.search(r"<id>(.*?)</id>", e)
            results.append(
                f"**{title.group(1).strip() if title else 'Unknown'}**\n"
                f"{link.group(1).strip() if link else ''}\n"
                f"{summary.group(1).strip()[:400] if summary else ''}"
            )
        return "\n\n---\n\n".join(results) or "No ArXiv results found."
    except Exception as e:
        return f"ArXiv error: {e}"


async def tool_search_semantic_scholar(query: str, http: httpx.AsyncClient) -> str:
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    for attempt in range(3):
        try:
            resp = await http.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": query, "limit": 3, "fields": "title,abstract,year,authors,url"},
                headers=headers,
                timeout=15.0,
            )
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                logger.warning("Semantic Scholar rate-limited — waiting %ds (attempt %d/3)", wait, attempt + 1)
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except httpx.HTTPStatusError:
            if attempt == 2:
                return "Semantic Scholar error: rate limit exceeded"
            continue
    try:
        papers = resp.json().get("data", [])
        if not papers:
            return "No Semantic Scholar results found."
        results = []
        for p in papers:
            authors = ", ".join(a["name"] for a in p.get("authors", [])[:3])
            results.append(
                f"**{p.get('title', '')}** ({p.get('year', '')})\n"
                f"Authors: {authors}\n{p.get('url', '')}\n"
                f"{(p.get('abstract') or '')[:400]}"
            )
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Semantic Scholar error: {e}"


async def tool_search_intel_kb(query: str, _http: httpx.AsyncClient) -> str:
    """Semantic search across all OSIA Qdrant collections."""
    try:
        from src.intelligence.qdrant_store import QdrantStore

        async with QdrantStore() as store:
            results = await store.cross_desk_search(query, top_k=5)
        # Filter low-confidence matches — cosine similarity below 0.45 is noise
        results = [r for r in results if r.score >= 0.45]
        if not results:
            return "No relevant intel found in the knowledge base for this query."
        parts = []
        for r in results:
            source = r.metadata.get("source", r.collection)
            date = r.metadata.get("collected_at", r.metadata.get("date", ""))
            header = f"[KB: {r.collection} | score={r.score:.2f} | source={source}"
            if date:
                header += f" | {date[:10]}"
            header += "]"
            parts.append(f"{header}\n{r.text[:600]}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"Intel KB search error: {e}"


async def tool_search_censys(query: str, http: httpx.AsyncClient) -> str:
    """Search Censys for internet-facing hosts, services, and TLS certificates."""
    if not CENSYS_API_ID or not CENSYS_API_SECRET:
        return "Censys unavailable: CENSYS_API_ID / CENSYS_API_SECRET not configured."
    try:
        resp = await http.get(
            "https://search.censys.io/api/v2/hosts/search",
            params={"q": query, "per_page": 5},
            auth=(CENSYS_API_ID, CENSYS_API_SECRET),
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("result", {}).get("hits", [])
        total = data.get("result", {}).get("total", 0)
        if not hits:
            return f"Censys: no results for '{query}'."
        results = []
        for h in hits:
            ip = h.get("ip", "")
            country = h.get("location", {}).get("country", "")
            asn_name = h.get("autonomous_system", {}).get("name", "")
            labels = ", ".join(h.get("labels", [])[:4])
            services = h.get("services", [])
            ports = ", ".join(f"{s.get('port')}/{s.get('service_name', '?')}" for s in services[:6])
            line = f"**{ip}** | AS: {asn_name} | Country: {country}"
            if ports:
                line += f" | Services: {ports}"
            if labels:
                line += f" | Labels: {labels}"
            results.append(line)
        return f"Censys — {total} total results for '{query}':\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"Censys error: {e}"


async def tool_search_criminalip(query: str, http: httpx.AsyncClient) -> str:
    """Search Criminal IP for threat-scored hosts, exposed services, and vulnerability data."""
    if not CRIMINALIP_API_KEY:
        return "Criminal IP unavailable: CRIMINALIP_API_KEY not configured."
    try:
        resp = await http.get(
            "https://api.criminalip.io/v1/asset/search",
            params={"query": query, "offset": 0},
            headers={"x-api-key": CRIMINALIP_API_KEY},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("data", {}).get("result", [])
        total = data.get("data", {}).get("count", 0)
        if not hits:
            return f"Criminal IP: no results for '{query}'."
        results = []
        for h in hits[:5]:
            ip = h.get("ip_address", "")
            country = h.get("country_code", "")
            org = h.get("org_name", "") or h.get("as_name", "")
            tags = ", ".join(h.get("tags", [])[:5])
            score = h.get("score", {})
            inbound = score.get("inbound", "")
            vuln_count = h.get("vulnerability_count", 0)
            ports_data = h.get("current_opened_port", {}).get("data", [])
            ports = ", ".join(str(p.get("port")) for p in ports_data[:6] if p.get("port"))
            line = f"**{ip}** | Org: {org} | Country: {country} | Threat: {inbound}"
            if vuln_count:
                line += f" | Vulns: {vuln_count}"
            if ports:
                line += f" | Ports: {ports}"
            if tags:
                line += f" | Tags: {tags}"
            results.append(line)
        return f"Criminal IP — {total} total results for '{query}':\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"Criminal IP error: {e}"


async def tool_search_otx(query: str, http: httpx.AsyncClient) -> str:
    """Search AlienVault OTX for threat intelligence pulses matching a query."""
    headers: dict[str, str] = {}
    if OTX_API_KEY:
        headers["X-OTX-API-KEY"] = OTX_API_KEY
    try:
        resp = await http.get(
            "https://otx.alienvault.com/api/v1/search/pulses",
            params={"q": query, "limit": 5, "sort": "-modified"},
            headers=headers,
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        pulses = data.get("results", [])
        if not pulses:
            return f"OTX: no threat intelligence pulses found for '{query}'."
        results = []
        for p in pulses:
            name = p.get("name", "")
            desc = (p.get("description", "") or "")[:300]
            tags = ", ".join(p.get("tags", [])[:8])
            indicator_count = p.get("indicators_count", 0)
            created = (p.get("created", "") or "")[:10]
            author = p.get("author", {}).get("username", "")
            entry = f"**{name}** (by {author}, {created})\nTags: {tags} | Indicators: {indicator_count}"
            if desc:
                entry += f"\n{desc}"
            results.append(entry)
        return f"OTX Threat Intelligence — results for '{query}':\n\n" + "\n\n---\n\n".join(results)
    except Exception as e:
        return f"AlienVault OTX error: {e}"


async def tool_search_aleph(query: str, http: httpx.AsyncClient) -> str:
    """Search OCCRP Aleph for entities across investigative journalism leak datasets (Panama Papers, FinCEN Files, Pandora Papers, etc.)."""
    headers: dict[str, str] = {"Accept": "application/json"}
    if ALEPH_API_KEY:
        headers["Authorization"] = f"ApiKey {ALEPH_API_KEY}"
    try:
        resp = await http.get(
            "https://aleph.occrp.org/api/2/entities",
            params={"q": query, "limit": 10, "filter:schemata": "Thing"},
            headers=headers,
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()
        entities = data.get("results", [])
        if not entities:
            return f"Aleph: no entities found for '{query}' in OCCRP investigative datasets."
        results = []
        for e in entities[:7]:
            caption = e.get("caption", "")
            schema = e.get("schema", "")
            dataset = e.get("dataset", {})
            dataset_label = dataset.get("label", dataset.get("name", "")) if isinstance(dataset, dict) else ""
            props = e.get("properties", {})
            notes = []
            for field in ("nationality", "country", "birthDate", "incorporationDate", "address", "registrationNumber"):
                vals = props.get(field, [])
                if vals:
                    notes.append(f"{field}: {', '.join(str(v) for v in vals[:2])}")
            entry = f"**{caption}** [{schema}] — Dataset: {dataset_label}"
            if notes:
                entry += "\n" + " | ".join(notes)
            results.append(entry)
        total = data.get("total", {})
        if isinstance(total, dict):
            total = total.get("value", len(entities))
        return f"OCCRP Aleph — {total} results for '{query}':\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"OCCRP Aleph error: {e}"


TOOL_REGISTRY = {
    "fetch_url": tool_fetch_url,
    "search_intel_kb": tool_search_intel_kb,
    "search_web": tool_search_web,
    "search_news": tool_search_news,
    "search_tavily": tool_search_tavily,
    "search_wikipedia": tool_search_wikipedia,
    "search_arxiv": tool_search_arxiv,
    "search_semantic_scholar": tool_search_semantic_scholar,
    "search_censys": tool_search_censys,
    "search_criminalip": tool_search_criminalip,
    "search_otx": tool_search_otx,
    "search_aleph": tool_search_aleph,
}

# OpenAI-format tool schemas for OpenRouter
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch and extract the full text of a web page or news article from a direct URL. "
                "Use when a search result returns a URL you want to read in full, or when a specific "
                "primary source URL is already known. Returns extracted article text (up to 15 000 chars). "
                "Do not call for social media, YouTube, or login-gated pages."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Full URL to fetch (http:// or https://)."}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_intel_kb",
            "description": (
                "Semantic search across all OSIA intelligence collections: desk reports, INTSUM archives, "
                "past research, MITRE ATT&CK, CVE database, CTI reports, TTP mappings, WikiLeaks cables, "
                "Epstein files, HackerOne disclosures, ICIJ Offshore Leaks (Panama Papers / Pandora Papers / "
                "Paradise Papers — 810K offshore entities, beneficial owners, shell companies), "
                "OFAC SDN sanctions list (18K+ sanctioned individuals and entities), "
                "Yahoo Finance news articles and earnings call transcripts. "
                "Always call this first — it may surface directly relevant existing intel and avoid redundant external queries."
            ),
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "DEFAULT web search via DuckDuckGo — no quota, use freely. "
                "Use for: background context, entity lookups, market data, product launches, living persons, "
                "and any general web query where recency is not the primary concern. "
                "Prefer search_news when you need dated recent articles."
            ),
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "DuckDuckGo NEWS search — returns recent articles with publication dates and source names. "
                "Prefer this over search_web when recency matters: breaking events, recent arrests or statements, "
                "current conflict updates, newly published reports, election results, sanctions announcements. "
                "No quota — use freely for any time-sensitive query."
            ),
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tavily",
            "description": (
                "PREMIUM web search via Tavily — limited monthly quota, use sparingly. "
                "Call ONLY when search_web results are clearly insufficient AND the query requires "
                "comprehensive, multi-source aggregation of breaking news from the last 24-48 hours. "
                "Never call if search_web already returned useful results for this query."
            ),
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": (
                "Wikipedia background lookup. "
                "Use for: established entities (orgs, countries, historical events, concepts, known persons). "
                "Skip for: breaking news, highly operational topics, financial data, niche technical specs."
            ),
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": (
                "ArXiv pre-print search. "
                "Use for: novel STEM research, ML/AI papers, cryptography, emerging technical concepts, economic research papers, monetary policy studies. "
                "Skip for: current events, politics, HUMINT, cultural topics."
            ),
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_semantic_scholar",
            "description": (
                "Semantic Scholar peer-reviewed literature search. "
                "Use for: established scientific topics requiring citation-heavy sourcing. "
                "Skip for: current events, HUMINT, geopolitics, finance, cyber threat intelligence."
            ),
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_otx",
            "description": (
                "Search AlienVault OTX (Open Threat Exchange) for community threat intelligence pulses. "
                "Use for: IOCs, malware campaigns, APT group activity, CVE exploitation reports, "
                "threat actor TTPs, recently reported C2 infrastructure. "
                "Returns pulse names, tags, indicator counts, and descriptions from the OTX community. "
                "Skip for: general news, HUMINT, geopolitics, or anything unrelated to cyber threats."
            ),
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_aleph",
            "description": (
                "Search OCCRP Aleph for entities in investigative journalism leak datasets: "
                "Panama Papers, Pandora Papers, FinCEN Files, Offshore Leaks, Russian Asset Tracker, "
                "EU lobbying registers, sanctions lists, and 300+ other datasets. "
                "Use for: corporate ownership, offshore entities, shell companies, PEP connections, "
                "sanctions exposure, financial crime leads, and subject background in leaked data. "
                "Skip for: general news, technical topics, or subjects with no financial/corporate angle."
            ),
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
]

# Per-desk guidance on which tools are most appropriate.
# Injected into the system prompt so the model makes informed choices upfront.
_DESK_TOOL_GUIDANCE: dict[str, str] = {
    "cyber-intelligence-and-warfare-desk": (
        "search_intel_kb FIRST (MITRE ATT&CK, CVE database, CTI reports, TTP mappings, HackerOne disclosures are all in the KB). "
        "search_otx for live threat intelligence pulses, IOCs, and campaign reporting when the topic involves an active threat actor or malware campaign. "
        "search_news for recent CVE disclosures, patch releases, or active exploitation reports. "
        "search_web for campaign reporting not covered by KB or OTX; use fetch_url to read full advisories. "
        "search_wikipedia for background on APT groups or malware families. "
        "search_arxiv only for novel malware research or cryptographic vulnerabilities. "
        "search_semantic_scholar is rarely appropriate. "
        "Reserve search_tavily only for breaking threat intelligence not yet indexed by DDG."
    ),
    "human-intelligence-and-profiling-desk": (
        "search_intel_kb FIRST (Epstein files, WikiLeaks cables, and past HUMINT research may surface direct leads), "
        "then search_aleph for the subject in OCCRP leak datasets (Panama Papers, FinCEN Files, Pandora Papers, sanctions lists). "
        "search_news for recent statements, arrests, appointments, or controversies involving the subject. "
        "search_web for current activity, affiliations, and background; use fetch_url to read full articles. "
        "search_wikipedia for background on subjects, organisations, or related events. "
        "Academic tools are not appropriate for profiling. "
        "Reserve search_tavily only if DDG results are clearly insufficient for a time-critical query."
    ),
    "cultural-and-theological-intelligence-desk": (
        "search_intel_kb FIRST (past cultural/theological research and Watch Floor INTSUMs may have relevant context), "
        "then search_wikipedia for deep doctrinal or historical background. "
        "search_news for recent events involving religious movements, cultural controversies, or clerical statements. "
        "search_web for current movements, recent events, or contemporary actors; use fetch_url to read primary sources. "
        "Academic tools only if the topic is specifically scholarly religious studies. "
        "search_tavily is rarely needed for cultural topics."
    ),
    "geopolitical-and-security-desk": (
        "search_intel_kb FIRST (WikiLeaks cables, past geopolitical INTSUMs, desk archives, and OFAC SDN sanctions list "
        "may have directly relevant intel on state actors, sanctioned individuals, or blocked entities). "
        "search_aleph when the topic involves oligarchs, sanctioned entities, or offshore financial networks. "
        "search_news for breaking conflict updates, diplomatic developments, and recent security events — results include "
        "publication dates so you can assess freshness. "
        "search_web for broader context; use fetch_url to read full reports. "
        "search_wikipedia for geopolitical context, state actors, historical background. "
        "Reserve search_tavily only for breaking conflict or sanctions news from the last 24-48h not indexed by DDG."
    ),
    "finance-and-economics-directorate": (
        "search_intel_kb FIRST — the KB contains ICIJ Offshore Leaks (Panama Papers, Pandora Papers, Paradise Papers: "
        "810K+ offshore entities, beneficial owners, shell company chains, intermediaries), "
        "OFAC SDN sanctions list (18K+ sanctioned individuals and entities with aliases, DOB, passports, addresses), "
        "Yahoo Finance news articles and earnings call transcripts, WikiLeaks cables, and Epstein files. "
        "search_aleph for additional corporate ownership, PEP connections, and leak datasets not yet in KB. "
        "search_news for recent earnings, market-moving events, regulatory actions, or corporate announcements. "
        "search_web for broader market context; use fetch_url to read full filings or articles. "
        "search_wikipedia for company or institution background. "
        "search_arxiv for economic research papers or monetary policy studies. "
        "Reserve search_tavily only for time-sensitive market-moving news DDG cannot surface."
    ),
    "science-technology-and-commercial-desk": (
        "search_intel_kb FIRST (past science/tech research may already cover this topic), "
        "then search_arxiv and search_semantic_scholar for technical research, patents, emerging science. "
        "search_news for recent product launches, commercial developments, or company announcements. "
        "search_web for broader context; use fetch_url to read full articles. "
        "search_wikipedia for established technical background. "
        "Reserve search_tavily only if DDG returns insufficient results for a rapidly evolving story."
    ),
    "information-warfare-desk": (
        "search_intel_kb FIRST (WikiLeaks cables and Epstein files may surface funding trails, "
        "personnel links, or documented covert influence programs directly relevant to the topic). "
        "search_news for active disinformation campaigns, recently published influence op reporting, "
        "or breaking media manipulation stories — dates are shown so you can assess currency. "
        "search_web for investigative journalism on propaganda and NGO funding disclosures; use fetch_url to read in full. "
        "search_wikipedia for background on organisations, historical psyops operations, or media ownership. "
        "Academic tools rarely needed unless the topic is formal propaganda studies or media theory. "
        "Reserve search_tavily only for rapidly developing disinfo campaigns not yet indexed by DDG."
    ),
    "environment-and-ecology-desk": (
        "search_news FIRST for current environmental events: extreme weather, disaster reports, crop failure, "
        "corporate pollution incidents, and official agency bulletins (NOAA, NASA, WMO, IPCC) — "
        "publication dates help confirm recency of fast-moving environmental stories. "
        "use fetch_url to read full agency reports or news articles. "
        "search_arxiv and search_semantic_scholar for climate science research, ecological studies, "
        "and peer-reviewed environmental impact assessments. "
        "search_intel_kb for past environmental intelligence, WikiLeaks cables on resource negotiations or "
        "corporate environmental lobbying. "
        "search_wikipedia for background on ecosystems, species, geographic regions, or corporations under investigation. "
        "Reserve search_tavily only for breaking disaster news in the last 24h that DDG has not indexed."
    ),
}

_REACT_TOOL_MAP = {
    "fetch_url": "fetch_url",
    "search_intel_kb": "search_intel_kb",
    "search_web": "search_web",
    "search_tavily": "search_tavily",
    "search_news": "search_news",
    "search_wikipedia": "search_wikipedia",
    "search_arxiv": "search_arxiv",
    "search_semantic_scholar": "search_semantic_scholar",
    "search_otx": "search_otx",
    "search_aleph": "search_aleph",
}


def _parse_react(text: str) -> list[tuple[str, str]]:
    """Extract (tool_name, query) pairs from ReAct-style model output."""
    calls = []
    for line in text.splitlines():
        for name in _REACT_TOOL_MAP:
            prefix = name.upper().replace("_", "_") + ":"
            if line.upper().startswith(prefix):
                query = line[len(prefix) :].strip()
                if query:
                    calls.append((name, query))
    return calls


# ---------------------------------------------------------------------------
# Query expansion — turn a bare topic + context into a focused research brief
# ---------------------------------------------------------------------------


async def _expand_research_query(job: ResearchJob, http: httpx.AsyncClient) -> str:
    """Build an enriched research prompt from the job's topic and available context.

    URL-sourced topics (triggered_by is a URL): instruct the model to fetch the
    source article first so it understands the specific angle before searching.

    All other topics: call a cheap model to produce a 1-2 sentence focus statement
    and 2-3 specific queries, replacing the bare ``Research topic: <word>`` message
    that leads to single-word tool calls.

    Falls back to plain ``Research topic: {topic}`` if all providers are unavailable.
    """
    triggered = job.triggered_by or ""
    source = job.source or ""

    # Build context lines for the expansion prompt.
    # For RSS-extracted topics triggered_by is the article URL — the article has
    # already been fetched and stored in Qdrant by the RSS ingress, so we pass
    # the URL as context for the expansion model only and never ask the research
    # model to re-fetch it.
    context_lines = [f"Topic: {job.topic}", f"Intelligence desk: {job.desk}"]
    if triggered.startswith(("http://", "https://")):
        context_lines.append(f"Extracted from article: {triggered}")
    elif triggered:
        context_lines.append(f"Triggered by: {triggered}")
    if source:
        context_lines.append(f"Source: {source}")

    expansion_prompt = (
        "You are an OSINT research planner. A bare topic name has been queued for intelligence research. "
        "Your job is to turn it into a precise, actionable research brief.\n\n"
        "Output exactly two lines (no extra text):\n"
        "FOCUS: <1-2 sentences on what specifically to investigate and why, given the desk and trigger context>\n"
        "QUERIES: <specific search query 1> | <specific search query 2> | <specific search query 3>\n\n"
        "Rules: queries must be specific phrases (4+ words), not single words or generic terms. "
        "Tailor them to the desk's intelligence domain.\n\n" + "\n".join(context_lines)
    )

    expansion: str | None = None

    if VENICE_API_KEY:
        try:
            resp = await http.post(
                f"{VENICE_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {VENICE_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": VENICE_MODEL_DEFAULT,
                    "messages": [{"role": "user", "content": expansion_prompt}],
                    "max_tokens": 200,
                    "temperature": 0.2,
                },
                timeout=20.0,
            )
            if resp.status_code == 200:
                expansion = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.debug("Query expansion (Venice) failed for '%s': %s", job.topic, exc)

    if not expansion and GEMINI_API_KEY:
        try:
            from google import genai

            _gemini = genai.Client(api_key=GEMINI_API_KEY)
            _resp = await asyncio.to_thread(
                _gemini.models.generate_content,
                model=GEMINI_MODEL,
                contents=expansion_prompt,
            )
            expansion = _resp.text.strip()
        except Exception as exc:
            logger.debug("Query expansion (Gemini) failed for '%s': %s", job.topic, exc)

    if expansion:
        logger.info("Expanded query for '%s': %s", job.topic, expansion.replace("\n", " ")[:160])
        return f"Research topic: {job.topic}\n\n{expansion}"

    # Fallback — no provider available or all failed
    return f"Research topic: {job.topic}"


# ---------------------------------------------------------------------------
# Message sanitisation
# ---------------------------------------------------------------------------


def _sanitize_assistant_message(msg: dict) -> dict:
    """Strip the assistant message to only fields Venice accepts.

    Venice's own API response includes extra fields (e.g. 'refusal': null) that its
    Pydantic validator then rejects when the message is sent back in a subsequent round,
    producing a cascade of union-type validation errors. Whitelist only the three
    fields that are always valid in an assistant turn.

    Also drops tool calls with empty/invalid function names — Venice occasionally
    emits these and then rejects its own output with a 400 on the next round.
    """
    cleaned: dict = {"role": "assistant", "content": msg.get("content") or ""}
    valid_calls = [tc for tc in (msg.get("tool_calls") or []) if tc.get("function", {}).get("name", "").strip()]
    if valid_calls:
        cleaned["tool_calls"] = valid_calls
    return cleaned


# ---------------------------------------------------------------------------
# OpenAI-compat research loop (Venice primary, OpenRouter fallback)
# ---------------------------------------------------------------------------


async def run_research_loop_openai_compat(
    job: ResearchJob,
    http: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    extra_headers: dict | None = None,
    model_override: str | None = None,
    user_message: str | None = None,
) -> str:
    model = model_override or _model_for_desk(job.desk)
    _registry = get_tool_registry(job.desk)  # desk-aware: search_intel_kb includes boost collections
    directives = (
        "\n\nAnalytical lens: Apply a decolonial and socialist materialist perspective. "
        "Prioritize labor rights, anti-imperialism, ecological impact, and data sovereignty. "
        "Avoid Washington Consensus framing."
        if job.directives_lens
        else ""
    )

    desk_tool_guidance = _DESK_TOOL_GUIDANCE.get(
        job.desk, "search_web for current context, search_wikipedia for background."
    )

    messages = [
        {
            "role": "system",
            "content": (
                f"You are an OSINT research analyst for the {job.desk}. "
                f"Produce a focused intelligence brief with citations.{directives}\n\n"
                "TOOL SELECTION — be deliberate and conservative:\n"
                "• search_intel_kb: ALWAYS call this first. Searches all OSIA collections (MITRE ATT&CK, CVEs, CTI, WikiLeaks, Epstein, ICIJ Offshore Leaks, OFAC sanctions, Yahoo Finance, past research). If KB results are sufficient, skip external tools.\n"
                "• search_news: DuckDuckGo NEWS search — returns recent articles with publication dates. Use when recency matters: breaking events, recent arrests, conflict updates, new reports.\n"
                "• search_web: general DuckDuckGo web search — no quota. Use for background context, entity lookups, and queries where recency is not the primary concern.\n"
                "• search_tavily: PREMIUM Tavily search — limited monthly budget. Call ONLY when search_web/search_news results are clearly insufficient AND you need multi-source breaking news from the last 24-48h.\n"
                "• fetch_url: fetch full article text from a known URL. Use after search results return relevant URLs.\n"
                "• search_wikipedia: background on established entities, concepts, or historical events.\n"
                "• search_arxiv: novel STEM/ML/security research papers. Not for news or politics.\n"
                "• search_semantic_scholar: peer-reviewed science. Not for HUMINT, cyber ops, or finance.\n\n"
                f"For this desk, preferred tools are: {desk_tool_guidance}\n\n"
                "Call only the tools that add distinct value for this specific topic. "
                "2-3 tool calls total is usually sufficient; stop as soon as you have enough to write the brief. "
                "Never run the same query through multiple tools.\n\n"
                "If native tool calling is unavailable, use the ReAct format:\n"
                "FETCH_URL: <url>\nSEARCH_INTEL_KB: <query>\nSEARCH_NEWS: <query>\nSEARCH_WEB: <query>\n"
                "SEARCH_TAVILY: <query>\nSEARCH_WIKIPEDIA: <query>\nSEARCH_ARXIV: <query>\n"
                "SEARCH_SEMANTIC_SCHOLAR: <query>\nSEARCH_CENSYS: <query>\nSEARCH_CRIMINALIP: <query>\n"
                "SEARCH_OTX: <query>\nSEARCH_ALEPH: <query>"
            ),
        },
        {"role": "user", "content": user_message or f"Research topic: {job.topic}"},
    ]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    logger.info("Researching: %s (desk: %s, model: %s)", job.topic, job.desk, model)

    for round_num in range(MAX_ROUNDS):
        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.3,
        }
        if model not in REACT_ONLY_MODELS:
            payload["tools"] = TOOL_SCHEMAS
            payload["tool_choice"] = "auto"

        for attempt in range(3):
            try:
                resp = await http.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120.0,
                )
                resp.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < 2:
                    wait = 35 * (attempt + 1)
                    logger.warning(
                        "429 rate-limited on round %d — waiting %ds (attempt %d/3)", round_num, wait, attempt + 1
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error(
                    "API HTTP %d on round %d — %s",
                    e.response.status_code,
                    round_num,
                    _http_error_detail(e),
                )
                raise
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError) as e:
                if attempt < 2:
                    wait = 15 * (attempt + 1)
                    logger.warning(
                        "Venice connection error on round %d — retrying in %ds (attempt %d/3): %s",
                        round_num,
                        wait,
                        attempt + 1,
                        e,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise

        data = resp.json()
        choice = data["choices"][0]
        message = choice["message"]
        sanitized = _sanitize_assistant_message(message)
        messages.append(sanitized)

        # Use the sanitized tool_calls — invalid names are already stripped,
        # so history and tool results stay consistent.
        tool_calls = sanitized.get("tool_calls") or []
        content = message.get("content", "") or ""

        # Native tool calls
        if tool_calls:
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}
                query = fn_args.get("query") or fn_args.get("url") or ""
                logger.info("Tool: %s(%r)", fn_name, query)
                tool_fn = _registry.get(fn_name)
                result = await tool_fn(query, http) if tool_fn else f"Unknown tool: {fn_name}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )
            continue

        # ReAct fallback — model emitted structured text instead of tool_calls
        react_calls = _parse_react(content)
        if react_calls:
            logger.info("Using ReAct fallback (%d calls)", len(react_calls))
            tool_results = []
            for fn_name, query in react_calls:
                logger.info("ReAct tool: %s(%r)", fn_name, query)
                tool_fn = _registry.get(fn_name)
                result = await tool_fn(query, http) if tool_fn else f"Unknown tool: {fn_name}"
                tool_results.append(f"[{fn_name}: {query}]\n{result}")
            messages.append({"role": "user", "content": "\n\n".join(tool_results)})
            continue

        # No tool calls and no ReAct — model is done
        logger.info("Research complete after %d rounds (%d chars)", round_num + 1, len(content))
        return content

    logger.warning("Hit max rounds for topic: %s", job.topic)
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            return m["content"]
    return ""


# ---------------------------------------------------------------------------
# Gemini fallback research loop
# ---------------------------------------------------------------------------


async def run_research_loop_gemini(job: ResearchJob, http: httpx.AsyncClient, user_message: str | None = None) -> str:
    from google import genai
    from google.genai import types

    gemini = genai.Client(api_key=GEMINI_API_KEY)
    _registry = get_tool_registry(job.desk)  # desk-aware: search_intel_kb includes boost collections
    directives = (
        "\n\nAnalytical lens: Apply a decolonial and socialist materialist perspective. "
        "Prioritize labor rights, anti-imperialism, ecological impact, and data sovereignty. "
        "Avoid Washington Consensus framing."
        if job.directives_lens
        else ""
    )

    tools = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="search_intel_kb",
                    description=(
                        "Semantic search across all OSIA intelligence collections: desk reports, INTSUM archives, "
                        "past research, MITRE ATT&CK, CVE database, CTI reports, TTP mappings, WikiLeaks cables, "
                        "Epstein files, HackerOne disclosures, ICIJ Offshore Leaks (Panama Papers / Pandora Papers / "
                        "Paradise Papers — 810K offshore entities and beneficial owners), "
                        "OFAC SDN sanctions list (18K+ sanctioned individuals and entities), "
                        "Yahoo Finance news and earnings call transcripts. Always call this first."
                    ),
                    parameters=types.Schema(
                        type="OBJECT", properties={"query": types.Schema(type="STRING")}, required=["query"]
                    ),
                ),
                types.FunctionDeclaration(
                    name="search_web",
                    description=(
                        "DEFAULT web search via DuckDuckGo — no quota, use freely. "
                        "Use for current events, recent news, background context, entity lookups, market data. "
                        "Prefer this over search_tavily for all routine queries."
                    ),
                    parameters=types.Schema(
                        type="OBJECT", properties={"query": types.Schema(type="STRING")}, required=["query"]
                    ),
                ),
                types.FunctionDeclaration(
                    name="search_tavily",
                    description=(
                        "PREMIUM web search via Tavily — limited monthly quota, use sparingly. "
                        "Call ONLY when search_web results are clearly insufficient AND the query requires "
                        "multi-source breaking news from the last 24-48 hours."
                    ),
                    parameters=types.Schema(
                        type="OBJECT", properties={"query": types.Schema(type="STRING")}, required=["query"]
                    ),
                ),
                types.FunctionDeclaration(
                    name="search_wikipedia",
                    description=(
                        "Wikipedia background lookup. "
                        "Use for established entities, historical events, concepts. "
                        "Skip for breaking news, financial data, niche operational topics."
                    ),
                    parameters=types.Schema(
                        type="OBJECT", properties={"query": types.Schema(type="STRING")}, required=["query"]
                    ),
                ),
                types.FunctionDeclaration(
                    name="search_arxiv",
                    description=(
                        "ArXiv pre-print search. "
                        "Use for novel STEM/ML/security research papers, economic research papers, and monetary policy studies. "
                        "Not for current events, politics, or HUMINT."
                    ),
                    parameters=types.Schema(
                        type="OBJECT", properties={"query": types.Schema(type="STRING")}, required=["query"]
                    ),
                ),
                types.FunctionDeclaration(
                    name="search_semantic_scholar",
                    description=(
                        "Semantic Scholar peer-reviewed literature. "
                        "Use for established scientific topics needing citation-heavy sourcing. "
                        "Not for HUMINT, cyber ops, geopolitics, or finance."
                    ),
                    parameters=types.Schema(
                        type="OBJECT", properties={"query": types.Schema(type="STRING")}, required=["query"]
                    ),
                ),
            ]
        )
    ]

    desk_tool_guidance = _DESK_TOOL_GUIDANCE.get(
        job.desk, "search_intel_kb first, then search_web for current context, search_wikipedia for background."
    )
    system = (
        f"You are an OSINT research analyst for the {job.desk}. "
        f"Produce a focused intelligence brief with citations.{directives}\n\n"
        "TOOL SELECTION — be deliberate and conservative. "
        "Always call search_intel_kb first — it searches all internal OSIA collections and may make external queries unnecessary. "
        f"For this desk, preferred tools are: {desk_tool_guidance} "
        "Call only the tools that add distinct value for this specific topic. "
        "2-3 tool calls total is usually sufficient. "
        "Never run the same query through multiple tools."
    )
    contents = [
        types.Content(
            role="user", parts=[types.Part(text=f"{system}\n\n{user_message or f'Research topic: {job.topic}'}")]
        )
    ]

    logger.info("Researching (Gemini): %s", job.topic)

    for round_num in range(MAX_ROUNDS):
        response = await asyncio.to_thread(
            gemini.models.generate_content,
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(tools=tools),
        )
        candidate = response.candidates[0]
        contents.append(candidate.content)

        function_calls = [p for p in candidate.content.parts if p.function_call]
        if not function_calls:
            text_parts = [p.text for p in candidate.content.parts if p.text]
            result = "\n".join(text_parts)
            logger.info("Research complete after %d rounds (%d chars)", round_num + 1, len(result))
            return result

        response_parts = []
        for part in function_calls:
            call = part.function_call
            _args = dict(call.args) if call.args else {}
            query = _args.get("query") or _args.get("url") or ""
            logger.info("Tool: %s(%r)", call.name, query)
            tool_fn = _registry.get(call.name)
            result_text = await tool_fn(query, http) if tool_fn else f"Unknown tool: {call.name}"
            response_parts.append(
                types.Part(function_response=types.FunctionResponse(name=call.name, response={"result": result_text}))
            )
        contents.append(types.Content(role="user", parts=response_parts))

    logger.warning("Hit max rounds for topic: %s", job.topic)
    text_parts = [p.text for p in contents[-1].parts if hasattr(p, "text") and p.text]
    return "\n".join(text_parts)


async def run_research_loop(job: ResearchJob, http: httpx.AsyncClient) -> str:
    """Try each configured provider in order; cascade to the next on any failure."""
    # Expand the bare topic into a focused brief + specific queries before dispatching.
    # The same expanded message is reused across provider retries so we only pay the
    # expansion cost once regardless of how many fallbacks are needed.
    user_message = await _expand_research_query(job, http)

    errors: list[str] = []

    if VENICE_API_KEY:
        try:
            return await run_research_loop_openai_compat(
                job, http, base_url=VENICE_BASE_URL, api_key=VENICE_API_KEY, user_message=user_message
            )
        except Exception as exc:
            logger.warning("Venice failed for '%s' — trying next provider: %s", job.topic, exc)
            errors.append(f"Venice: {exc}")

    if OPENROUTER_API_KEY:
        try:
            return await run_research_loop_openai_compat(
                job,
                http,
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
                extra_headers={"HTTP-Referer": "https://osia.dev", "X-Title": "OSIA Research Worker"},
                model_override=OPENROUTER_RESEARCH_MODEL,
                user_message=user_message,
            )
        except Exception as exc:
            logger.warning("OpenRouter failed for '%s' — trying Gemini: %s", job.topic, exc)
            errors.append(f"OpenRouter: {exc}")

    if GEMINI_API_KEY:
        try:
            return await run_research_loop_gemini(job, http, user_message=user_message)
        except Exception as exc:
            logger.warning("Gemini failed for '%s': %s", job.topic, exc)
            errors.append(f"Gemini: {exc}")

    raise RuntimeError(f"All providers failed for '{job.topic}': {'; '.join(errors) or 'no API keys configured'}")


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

# Minimum prose length for a result to be worth storing.
# 500 chars is roughly two short paragraphs — anything under is either a
# model failure or raw tool-call artifacts rather than an intelligence brief.
_MIN_RESEARCH_CHARS = 500

# ReAct line prefixes that indicate the text is an un-executed tool call
# artifact rather than synthesised prose.
_ARTIFACT_PREFIXES = tuple(
    prefix for name in _REACT_TOOL_MAP for prefix in (f"{name.upper()}:", f"{name}{{", f"{name}(")
)


def _validate_research_output(text: str, topic: str) -> str | None:
    """Return the stripped text if it qualifies as a usable intelligence brief.

    Rejects:
    - Outputs shorter than _MIN_RESEARCH_CHARS (model gave up or errored)
    - Outputs where stripping known ReAct tool-call lines leaves less than
      300 chars of actual prose (raw artifact stored instead of a brief)
    """
    stripped = text.strip()

    if len(stripped) < _MIN_RESEARCH_CHARS:
        logger.warning(
            "Research output too short (%d chars) for '%s' — discarding",
            len(stripped),
            topic,
        )
        return None

    prose_lines = [
        ln
        for ln in stripped.splitlines()
        if not any(ln.upper().startswith(p) or ln.startswith(p.lower()) for p in _ARTIFACT_PREFIXES)
    ]
    prose = "\n".join(prose_lines).strip()
    if len(prose) < 300:
        logger.warning(
            "Research output for '%s' appears to be raw tool-call artifacts (%d chars prose) — discarding",
            topic,
            len(prose),
        )
        return None

    return stripped


# ---------------------------------------------------------------------------
# Chunking and storage
# ---------------------------------------------------------------------------


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    overlap = CHUNK_SIZE // 5
    step = CHUNK_SIZE - overlap
    return [" ".join(words[i : i + CHUNK_SIZE]) for i in range(0, len(words), step) if words[i : i + CHUNK_SIZE]]


_WIKI_STOPWORDS = frozenset(
    "the a an and or of in on at to for with by from is are was were be been "
    "has have had its it this that these those".split()
)


def _wiki_match_entity(topic: str, results: list[dict]) -> dict | None:
    """Return the first entity search result that is genuinely about the topic.

    Wiki full-text search returns relevance-ranked results but can match on
    incidental word overlap (e.g. 'polar bears' matching a page that previously
    received a 'bear' brief). We require that at least one significant token from
    the topic appears in the result's title or slug to confirm relevance.
    """
    topic_tokens = {
        t.lower().strip("\"'.,;:()") for t in topic.split() if len(t) > 2 and t.lower() not in _WIKI_STOPWORDS
    }
    for r in results:
        if not r.get("path", "").startswith("entities/"):
            continue
        title = r.get("title", "").lower()
        slug = r.get("path", "").split("/")[-1].replace("-", " ")
        candidate_text = f"{title} {slug}"
        if any(tok in candidate_text for tok in topic_tokens):
            return r
    return None


# Summary content considered a stub — safe to replace with new research.
_WIKI_STUB_MARKERS = (
    "pending research worker analysis",
    "pending hermes corroboration",
    "summary pending",
    "no summary compiled",
)
# Minimum chars for an existing summary to be considered substantive.
_WIKI_SUMMARY_MIN_CHARS = 500


def _wiki_extract_section(content: str, section: str) -> str:
    """Return the inner text of an OSIA:AUTO-fenced section, or empty string."""
    m = re.search(
        rf"<!-- OSIA:AUTO:{re.escape(section)} -->(.*?)<!-- /OSIA:AUTO:{re.escape(section)} -->",
        content,
        re.DOTALL,
    )
    return m.group(1).strip() if m else ""


def _wiki_replace_section(content: str, section: str, new_inner: str) -> str:
    """Replace the inner text of an OSIA:AUTO-fenced section in place."""
    open_tag = f"<!-- OSIA:AUTO:{section} -->"
    close_tag = f"<!-- /OSIA:AUTO:{section} -->"
    return re.sub(
        rf"{re.escape(open_tag)}.*?{re.escape(close_tag)}",
        f"{open_tag}\n{new_inner.strip()}\n{close_tag}",
        content,
        flags=re.DOTALL,
    )


def _wiki_append_section(content: str, section: str, new_entry: str) -> str:
    """Append an entry to an OSIA:AUTO-fenced section in place."""
    open_tag = f"<!-- OSIA:AUTO:{section} -->"
    close_tag = f"<!-- /OSIA:AUTO:{section} -->"

    def _do_append(m: re.Match) -> str:
        inner = m.group(1).rstrip()
        return f"{open_tag}\n{inner}\n{new_entry.strip()}\n{close_tag}"

    return re.sub(
        rf"{re.escape(open_tag)}(.*?){re.escape(close_tag)}",
        _do_append,
        content,
        flags=re.DOTALL,
    )


async def _wiki_append_research_note(job: ResearchJob, text: str, http: httpx.AsyncClient) -> None:
    """Write research output to the entity wiki page for job.topic.

    Enriching an existing page:
      - Fetches the page once, modifies both sections in memory, writes once.
      - Only replaces the summary if it is a stub/placeholder or shorter than
        _WIKI_SUMMARY_MIN_CHARS — preserves substantive existing intel.
      - Always appends a dated entry to research-notes.

    Creating a new page:
      - Builds a full entity page with the research text as the initial summary.

    Non-fatal — logs and returns on any failure.
    """
    if not os.getenv("WIKIJS_API_KEY"):
        return
    try:
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        desk_section = desk_wiki_section(job.desk) if job.desk else "desks"
        desk_label = job.desk or "research-worker"

        excerpt = text[:300].replace("\n", " ").strip()
        if len(text) > 300:
            excerpt += "…"
        note = f"- **{date_str}** — [{desk_label}](/{desk_section}) research on **{job.topic}**\n\n  > {excerpt}"

        async with WikiClient(http) as wiki:
            results = await wiki.search_pages(job.topic[:50])
            match = _wiki_match_entity(job.topic, results)

            if match:
                page = await wiki.get_page(match["path"])
                if not page:
                    # Disappeared between search and fetch — fall through to create
                    match = None
                else:
                    body = page["content"]
                    existing_summary = _wiki_extract_section(body, "summary")

                    is_stub = len(existing_summary) < _WIKI_SUMMARY_MIN_CHARS or any(
                        marker in existing_summary.lower() for marker in _WIKI_STUB_MARKERS
                    )

                    if is_stub:
                        body = _wiki_replace_section(body, "summary", text.strip())
                        action = "summary+notes"
                    else:
                        action = "notes-only"

                    body = _wiki_append_section(body, "research-notes", note)

                    await wiki.update_page(
                        page["id"],
                        body,
                        page["title"],
                        page.get("description", ""),
                        page.get("tags", []),
                    )
                    logger.info("Wiki: enriched entity page '%s' (%s)", match["path"], action)

            if not match:
                entity_type = job.entity_type or "Organisation"
                guess_path = entity_wiki_path(entity_type, job.topic)
                content = build_entity_page(
                    entity_type=entity_type,
                    desk_name=desk_label,
                    desk_section=desk_section,
                    first_seen=date_str,
                    summary=text.strip(),
                )
                created = await wiki.create_page(
                    guess_path,
                    job.topic,
                    content,
                    description=f"Entity — first researched {date_str} by {desk_label}",
                    tags=["entity", entity_type.lower(), "research-worker"],
                )
                if created:
                    logger.info("Wiki: created entity page '%s'", guess_path)
    except Exception as e:
        logger.warning("Wiki research note update failed (non-fatal): %s", e)


# Entity types that are expected for each desk. Mismatches are logged as
# warnings so contaminated points can be audited — storage is not blocked.
_DESK_ENTITY_ALIGNMENT: dict[str, frozenset[str]] = {
    "human-intelligence-and-profiling-desk": frozenset({"Person"}),
    "geopolitical-and-security-desk": frozenset({"Organisation", "Location", "Event", "Concept"}),
    "cyber-intelligence-and-warfare-desk": frozenset({"Technology", "Organisation"}),
    "cultural-and-theological-intelligence-desk": frozenset({"Organisation", "Person", "Concept"}),
    "science-technology-and-commercial-desk": frozenset({"Technology", "Organisation"}),
    "finance-and-economics-directorate": frozenset({"Organisation", "Person"}),
    "information-warfare-desk": frozenset({"Organisation", "Person", "Event"}),
    "environment-and-ecology-desk": frozenset({"Organisation", "Location", "Event"}),
}


async def store_research(job: ResearchJob, text: str, http: httpx.AsyncClient, qdrant: QdrantClient):
    chunks = _chunk_text(text)
    if not chunks:
        logger.warning("No chunks for job %s", job.job_id)
        return

    # Warn if the entity type doesn't match the desk's expected domain — these
    # points may contaminate RAG results and should be flagged for periodic audit.
    if job.entity_type and job.desk in _DESK_ENTITY_ALIGNMENT:
        if job.entity_type not in _DESK_ENTITY_ALIGNMENT[job.desk]:
            logger.warning(
                "Desk alignment mismatch: entity_type=%r stored in %s (expected %s) — topic=%r",
                job.entity_type,
                job.desk,
                sorted(_DESK_ENTITY_ALIGNMENT[job.desk]),
                job.topic,
            )

    embeddings = await embed_texts(chunks, http)
    now = datetime.now(UTC).isoformat()
    now_unix = int(time.time())
    collection = job.desk if job.desk else RESEARCH_COLLECTION
    # Entity tags: the researched topic itself + individual tokens for filter matching
    entity_tags = list({job.topic} | {t for t in job.topic.split() if len(t) > 3})
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=False)):
        point_id = int(hashlib.md5(f"{job.job_id}:{i}".encode()).hexdigest()[:8], 16)  # noqa: S324
        points.append(
            {
                "id": point_id,
                "vector": vector,
                "payload": {
                    "text": chunk,
                    "topic": job.topic,
                    "desk": job.desk,
                    "entity_type": job.entity_type or "",
                    "job_id": job.job_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "triggered_by": job.triggered_by,
                    "collected_at": now,
                    "ingested_at_unix": now_unix,
                    "source": "research_worker",
                    "reliability_tier": "B",
                    "entity_tags": entity_tags,
                },
            }
        )
    await qdrant.ensure_collection(collection)
    await qdrant.upsert_points(points, collection)


# ---------------------------------------------------------------------------
# Main — oneshot batch: drain queue, process all, exit
# ---------------------------------------------------------------------------


async def main():
    logger.info("=== OSIA Research Worker starting ===")

    if not VENICE_API_KEY and not OPENROUTER_API_KEY and not GEMINI_API_KEY:
        logger.error("No API key set (VENICE_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY) — cannot run")
        return

    if VENICE_API_KEY:
        backend = f"Venice ({VENICE_MODEL_UNCENSORED} / {VENICE_MODEL_CYBER} / {VENICE_MODEL_DEFAULT})"
    elif OPENROUTER_API_KEY:
        backend = "OpenRouter (fallback)"
    else:
        backend = "Gemini (fallback)"
    logger.info("Backend: %s | Qdrant: %s", backend, QDRANT_URL)

    queue = RedisQueue()

    depth = queue.depth()
    logger.info("Research queue depth: %d (threshold: %d)", depth, BATCH_THRESHOLD)

    if depth < BATCH_THRESHOLD:
        logger.info("Queue below threshold — nothing to do.")
        return

    async with httpx.AsyncClient(timeout=60.0) as http:
        qdrant = QdrantClient(http)

        succeeded = 0
        failed = 0
        skipped = 0

        for _ in range(MAX_JOBS_PER_RUN):
            result = queue.pop()
            if result is None:
                break
            payload, raw = result

            job = ResearchJob.from_dict(payload)
            if not job.topic:
                queue.complete(raw)
                continue

            topic_key = hashlib.md5(job.topic.lower().strip().encode()).hexdigest()  # noqa: S324
            if queue.is_seen(topic_key):
                logger.info("Already researched, skipping: %s", job.topic)
                queue.complete(raw)
                skipped += 1
                continue

            t0 = time.monotonic()
            try:
                text = await run_research_loop(job, http)
            except Exception as e:
                logger.error("Research failed for '%s': %s", job.topic, e)
                queue.complete(raw)
                queue.release_inflight(job.topic)
                failed += 1
                continue

            text = _validate_research_output(text or "", job.topic)
            if not text:
                queue.complete(raw)
                queue.release_inflight(job.topic)
                failed += 1
                continue

            logger.info("Research done in %.1fs (%d chars): %s", time.monotonic() - t0, len(text), job.topic)

            try:
                await store_research(job, text, http, qdrant)
            except Exception as e:
                logger.error("Storage failed for '%s': %s", job.topic, e)
                queue.complete(raw)
                queue.release_inflight(job.topic)
                failed += 1
                continue

            await _wiki_append_research_note(job, text, http)
            queue.mark_seen(topic_key, topic=job.topic)
            queue.complete(raw)
            succeeded += 1
            await asyncio.sleep(2)  # brief pause between jobs to stay within Venice rate limits

        logger.info(
            "=== Batch complete. succeeded=%d failed=%d skipped=%d | queue depth: %d ===",
            succeeded,
            failed,
            skipped,
            queue.depth(),
        )


if __name__ == "__main__":
    asyncio.run(main())
