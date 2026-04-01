"""
OSIA Research Worker — oneshot batch processor, runs locally via systemd timer.

Drains osia:research_queue, runs a multi-turn tool-calling research loop via
Venice AI (uncensored/permissive model routing per desk), then chunks and
embeds results into Qdrant for retrieval-augmented generation at report time.

Desk → model routing:
  HUMINT / Cultural / Geopolitical / InfoWar  → venice-uncensored          (no guardrails, ReAct tool use)
  Cyber / Finance                             → mistral-31-24b             (Venice-private, native FC)
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
  RESEARCH_COOLDOWN_HOURS     — Hours before a topic can be re-researched (default: 72)
  VENICE_MODEL_UNCENSORED     — Override uncensored model slug (default: venice-uncensored)
  VENICE_MODEL_CYBER          — Override cyber desk model slug (default: mistral-31-24b)
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

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("osia.research_worker")

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

WIKIPEDIA_USER_AGENT = os.getenv(
    "WIKIPEDIA_USER_AGENT",
    "OSIA-Intelligence-Framework/1.0 (https://osia.dev; research@osia.dev) python-httpx",
)

BATCH_THRESHOLD = int(os.getenv("RESEARCH_BATCH_THRESHOLD", "3"))
RESEARCH_COOLDOWN_SECONDS = int(os.getenv("RESEARCH_COOLDOWN_HOURS", "72")) * 3600

# Venice model routing per desk
VENICE_MODEL_UNCENSORED = os.getenv("VENICE_MODEL_UNCENSORED", "venice-uncensored")
VENICE_MODEL_CYBER = os.getenv("VENICE_MODEL_CYBER", "mistral-31-24b")
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


def _model_for_desk(desk: str) -> str:
    if desk in UNCENSORED_DESKS:
        return VENICE_MODEL_UNCENSORED
    if desk in CYBER_DESKS:
        return VENICE_MODEL_CYBER
    if desk in FINANCE_DESKS:
        return VENICE_MODEL_CYBER  # mistral-31-24b — same tier as Cyber
    return VENICE_MODEL_DEFAULT


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ResearchJob:
    job_id: str
    topic: str
    desk: str
    priority: str = "normal"
    directives_lens: bool = True
    triggered_by: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchJob":
        return cls(
            job_id=d.get("job_id", str(uuid.uuid4())),
            topic=d.get("topic", ""),
            desk=d.get("desk", "collection-directorate"),
            priority=d.get("priority", "normal"),
            directives_lens=d.get("directives_lens", True),
            triggered_by=d.get("triggered_by", ""),
        )


# ---------------------------------------------------------------------------
# Redis client (direct — no HTTP queue API needed when running locally)
# ---------------------------------------------------------------------------


class RedisQueue:
    def __init__(self):
        import redis

        self._r = redis.from_url(REDIS_URL, decode_responses=True)

    def depth(self) -> int:
        return self._r.llen("osia:research_queue")

    def pop(self) -> dict | None:
        result = self._r.blpop("osia:research_queue", timeout=2)
        if result is None:
            return None
        _, raw = result
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def is_seen(self, key: str) -> bool:
        return bool(self._r.exists(f"osia:research:seen:{key}"))

    def mark_seen(self, key: str):
        self._r.set(f"osia:research:seen:{key}", "1", ex=RESEARCH_COOLDOWN_SECONDS)


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


async def tool_search_duckduckgo(query: str, _http: httpx.AsyncClient) -> str:
    """DuckDuckGo web search — no API key required."""
    try:
        from duckduckgo_search import DDGS

        results = await asyncio.to_thread(lambda: list(DDGS().text(query, max_results=5)))
        if not results:
            return "No DuckDuckGo results found."
        return "\n\n".join(f"[{r.get('title', '')}]({r.get('href', '')})\n{r.get('body', '')[:500]}" for r in results)
    except Exception as e:
        return f"DuckDuckGo search error: {e}"


async def tool_search_web(query: str, http: httpx.AsyncClient) -> str:
    """Live web search via Tavily (primary) with automatic DuckDuckGo fallback."""
    if not TAVILY_API_KEY:
        logger.info("TAVILY_API_KEY not set — using DuckDuckGo for web search")
        return await tool_search_duckduckgo(query, http)
    try:
        resp = await http.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 5},
            timeout=15.0,
        )
        if resp.status_code == 432:
            logger.warning("Tavily quota exhausted (432) — falling back to DuckDuckGo")
            return await tool_search_duckduckgo(query, http)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return "\n\n".join(f"[{r.get('title', '')}]({r.get('url', '')})\n{r.get('content', '')[:500]}" for r in results)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 432:
            logger.warning("Tavily quota exhausted — falling back to DuckDuckGo")
            return await tool_search_duckduckgo(query, http)
        logger.warning("Tavily HTTP error %d — falling back to DuckDuckGo: %s", e.response.status_code, e)
        return await tool_search_duckduckgo(query, http)
    except Exception as e:
        logger.warning("Tavily error — falling back to DuckDuckGo: %s", e)
        return await tool_search_duckduckgo(query, http)


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

        store = QdrantStore()
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
    "search_intel_kb": tool_search_intel_kb,
    "search_web": tool_search_web,
    "search_duckduckgo": tool_search_duckduckgo,
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
                "Live web search (Tavily → DuckDuckGo fallback). "
                "Use for: current events, recent news, threat activity, market data, product launches, living persons. "
                "Skip for: well-established facts, historical topics, academic literature."
            ),
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_duckduckgo",
            "description": (
                "DuckDuckGo web search — no quota. "
                "Use ONLY if search_web already failed or returned poor results for this query. "
                "Never call both search_web and search_duckduckgo for the same query."
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
        "search_web for recent CVE news or campaign reporting not covered by KB or OTX. "
        "search_wikipedia for background on APT groups or malware families. "
        "search_arxiv only for novel malware research or cryptographic vulnerabilities. "
        "search_semantic_scholar is rarely appropriate."
    ),
    "human-intelligence-and-profiling-desk": (
        "search_intel_kb FIRST (Epstein files, WikiLeaks cables, and past HUMINT research may surface direct leads), "
        "then search_aleph for the subject in OCCRP leak datasets (Panama Papers, FinCEN Files, Pandora Papers, sanctions lists). "
        "search_web for current activity, affiliations, recent statements. "
        "search_wikipedia for background on subjects, organisations, or related events. "
        "Academic tools are not appropriate for profiling."
    ),
    "cultural-and-theological-intelligence-desk": (
        "search_intel_kb FIRST (past cultural/theological research and Watch Floor INTSUMs may have relevant context), "
        "then search_wikipedia for deep doctrinal or historical background. "
        "search_web for current movements, recent events, or contemporary actors. "
        "Academic tools only if the topic is specifically scholarly religious studies."
    ),
    "geopolitical-and-security-desk": (
        "search_intel_kb FIRST (WikiLeaks cables, past geopolitical INTSUMs, desk archives, and OFAC SDN sanctions list "
        "may have directly relevant intel on state actors, sanctioned individuals, or blocked entities). "
        "search_aleph when the topic involves oligarchs, sanctioned entities, or offshore financial networks. "
        "search_web for current events, diplomatic developments, conflict reporting. "
        "search_wikipedia for geopolitical context, state actors, historical background. "
        "Academic tools rarely needed unless the topic involves strategic theory or sanctions research."
    ),
    "finance-and-economics-directorate": (
        "search_intel_kb FIRST — the KB contains ICIJ Offshore Leaks (Panama Papers, Pandora Papers, Paradise Papers: "
        "810K+ offshore entities, beneficial owners, shell company chains, intermediaries), "
        "OFAC SDN sanctions list (18K+ sanctioned individuals and entities with aliases, DOB, passports, addresses), "
        "Yahoo Finance news articles and earnings call transcripts, WikiLeaks cables, and Epstein files. "
        "search_aleph for additional corporate ownership, PEP connections, and leak datasets not yet in KB. "
        "search_web for market news, economic indicators, corporate reporting, recent sanctions updates. "
        "search_wikipedia for company or institution background. "
        "search_arxiv for economic research papers or monetary policy studies."
    ),
    "science-technology-and-commercial-desk": (
        "search_intel_kb FIRST (past science/tech research may already cover this topic), "
        "then search_arxiv and search_semantic_scholar for technical research, patents, emerging science. "
        "search_web for recent product launches, commercial developments, company news. "
        "Wikipedia for established technical background."
    ),
    "information-warfare-desk": (
        "search_intel_kb FIRST (WikiLeaks cables and Epstein files may surface funding trails, "
        "personnel links, or documented covert influence programs directly relevant to the topic). "
        "search_web for current influence operations, disinformation campaign reporting, NGO funding disclosures, "
        "and academic or investigative journalism on propaganda and psyops. "
        "search_wikipedia for background on organisations, historical psyops operations, or media ownership. "
        "Academic tools rarely needed unless the topic is formal propaganda studies or media theory."
    ),
    "environment-and-ecology-desk": (
        "search_web FIRST for current environmental events: extreme weather, disaster reports, crop failure news, "
        "corporate pollution incidents, ecocide reporting, and official agency bulletins (NOAA, NASA, WMO, IPCC). "
        "search_arxiv and search_semantic_scholar for climate science research, ecological studies, "
        "and peer-reviewed environmental impact assessments. "
        "search_intel_kb for past environmental intelligence, WikiLeaks cables on resource negotiations or "
        "corporate environmental lobbying. "
        "search_wikipedia for background on ecosystems, species, geographic regions, or corporations under investigation."
    ),
}

_REACT_TOOL_MAP = {
    "search_intel_kb": "search_intel_kb",
    "search_web": "search_web",
    "search_duckduckgo": "search_duckduckgo",
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
# OpenAI-compat research loop (Venice primary, OpenRouter fallback)
# ---------------------------------------------------------------------------


async def run_research_loop_openai_compat(
    job: ResearchJob,
    http: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    extra_headers: dict | None = None,
) -> str:
    model = _model_for_desk(job.desk)
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
                "• search_web: current events, recent news, live threat data, market updates.\n"
                "• search_duckduckgo: web fallback only — do NOT call if search_web already ran for the same query.\n"
                "• search_wikipedia: background on established entities, concepts, or historical events.\n"
                "• search_arxiv: novel STEM/ML/security research papers. Not for news or politics.\n"
                "• search_semantic_scholar: peer-reviewed science. Not for HUMINT, cyber ops, or finance.\n\n"
                f"For this desk, preferred tools are: {desk_tool_guidance}\n\n"
                "Call only the tools that add distinct value for this specific topic. "
                "2-3 tool calls total is usually sufficient; stop as soon as you have enough to write the brief. "
                "Never run the same query through multiple tools.\n\n"
                "If native tool calling is unavailable, use the ReAct format:\n"
                "SEARCH_INTEL_KB: <query>\nSEARCH_WEB: <query>\nSEARCH_DUCKDUCKGO: <query>\n"
                "SEARCH_WIKIPEDIA: <query>\nSEARCH_ARXIV: <query>\nSEARCH_SEMANTIC_SCHOLAR: <query>\n"
                "SEARCH_OTX: <query>\nSEARCH_ALEPH: <query>"
            ),
        },
        {"role": "user", "content": f"Research topic: {job.topic}"},
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
                    "API HTTP %d on round %d — body: %s",
                    e.response.status_code,
                    round_num,
                    e.response.text[:500],
                )
                raise

        data = resp.json()
        choice = data["choices"][0]
        message = choice["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        content = message.get("content", "") or ""

        # Native tool calls
        if tool_calls:
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}
                query = fn_args.get("query", "")
                logger.info("Tool: %s(%r)", fn_name, query)
                tool_fn = TOOL_REGISTRY.get(fn_name)
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
                tool_fn = TOOL_REGISTRY.get(fn_name)
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


async def run_research_loop_gemini(job: ResearchJob, http: httpx.AsyncClient) -> str:
    from google import genai
    from google.genai import types

    gemini = genai.Client(api_key=GEMINI_API_KEY)
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
                        "Live web search (Tavily → DuckDuckGo fallback). "
                        "Use for current events, recent news, threat activity, market data. "
                        "Skip for well-established facts or academic literature."
                    ),
                    parameters=types.Schema(
                        type="OBJECT", properties={"query": types.Schema(type="STRING")}, required=["query"]
                    ),
                ),
                types.FunctionDeclaration(
                    name="search_duckduckgo",
                    description=(
                        "DuckDuckGo web search — no quota. "
                        "Use ONLY if search_web already failed for this query. "
                        "Never call both search_web and search_duckduckgo for the same query."
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
    contents = [types.Content(role="user", parts=[types.Part(text=f"{system}\n\nResearch topic: {job.topic}")])]

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
            query = dict(call.args).get("query", "") if call.args else ""
            logger.info("Tool: %s(%r)", call.name, query)
            tool_fn = TOOL_REGISTRY.get(call.name)
            result_text = await tool_fn(query, http) if tool_fn else f"Unknown tool: {call.name}"
            response_parts.append(
                types.Part(function_response=types.FunctionResponse(name=call.name, response={"result": result_text}))
            )
        contents.append(types.Content(role="user", parts=response_parts))

    logger.warning("Hit max rounds for topic: %s", job.topic)
    text_parts = [p.text for p in contents[-1].parts if hasattr(p, "text") and p.text]
    return "\n".join(text_parts)


async def run_research_loop(job: ResearchJob, http: httpx.AsyncClient) -> str:
    if VENICE_API_KEY:
        return await run_research_loop_openai_compat(job, http, base_url=VENICE_BASE_URL, api_key=VENICE_API_KEY)
    if OPENROUTER_API_KEY:
        logger.warning("VENICE_API_KEY not set — falling back to OpenRouter")
        return await run_research_loop_openai_compat(
            job,
            http,
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            extra_headers={"HTTP-Referer": "https://osia.dev", "X-Title": "OSIA Research Worker"},
        )
    if GEMINI_API_KEY:
        logger.warning("No Venice/OpenRouter key — falling back to Gemini (may refuse sensitive topics)")
        return await run_research_loop_gemini(job, http)
    raise RuntimeError("No API key set: VENICE_API_KEY, OPENROUTER_API_KEY, or GEMINI_API_KEY required")


# ---------------------------------------------------------------------------
# Chunking and storage
# ---------------------------------------------------------------------------


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    overlap = CHUNK_SIZE // 5
    step = CHUNK_SIZE - overlap
    return [" ".join(words[i : i + CHUNK_SIZE]) for i in range(0, len(words), step) if words[i : i + CHUNK_SIZE]]


async def store_research(job: ResearchJob, text: str, http: httpx.AsyncClient, qdrant: QdrantClient):
    chunks = _chunk_text(text)
    if not chunks:
        logger.warning("No chunks for job %s", job.job_id)
        return
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
                    "job_id": job.job_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "triggered_by": job.triggered_by,
                    "collected_at": now,
                    "ingested_at_unix": now_unix,
                    "source": "research_worker",
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

        # Drain the queue
        jobs: list[ResearchJob] = []
        while True:
            payload = queue.pop()
            if payload is None:
                break
            job = ResearchJob.from_dict(payload)
            if job.topic:
                jobs.append(job)

        logger.info("Drained %d jobs from queue", len(jobs))
        if not jobs:
            return

        succeeded = 0
        failed = 0
        skipped = 0

        for job in jobs:
            topic_key = hashlib.md5(job.topic.lower().strip().encode()).hexdigest()  # noqa: S324
            if queue.is_seen(topic_key):
                logger.info("Already researched, skipping: %s", job.topic)
                skipped += 1
                continue

            t0 = time.monotonic()
            try:
                text = await run_research_loop(job, http)
            except Exception as e:
                logger.error("Research failed for '%s': %s", job.topic, e)
                failed += 1
                continue

            if not text:
                logger.warning("Empty result for topic: %s", job.topic)
                failed += 1
                continue

            logger.info("Research done in %.1fs (%d chars): %s", time.monotonic() - t0, len(text), job.topic)

            try:
                await store_research(job, text, http, qdrant)
            except Exception as e:
                logger.error("Storage failed for '%s': %s", job.topic, e)
                failed += 1
                continue

            queue.mark_seen(topic_key)
            succeeded += 1
            await asyncio.sleep(2)  # brief pause between jobs to stay within Venice rate limits

    logger.info("=== Batch complete. succeeded=%d failed=%d skipped=%d ===", succeeded, failed, skipped)


if __name__ == "__main__":
    asyncio.run(main())
