"""
OSIA Research Worker — oneshot batch processor, runs locally via systemd timer.

Drains osia:research_queue, runs a multi-turn tool-calling research loop via
Venice AI (uncensored/permissive model routing per desk), then chunks and
embeds results into Qdrant for retrieval-augmented generation at report time.

Desk → model routing:
  HUMINT / Cultural / Geopolitical  → venice-uncensored (no guardrails, ReAct tool use)
  Cyber                             → mistral-31-24b    (Venice-private, native FC)
  Finance / Science / default       → mistral-small-3-2-24b-instruct (cheap, native FC)

Fallback chain: Venice → OpenRouter → Gemini

Environment variables:
  VENICE_API_KEY              — Venice API key (primary)
  OPENROUTER_API_KEY          — OpenRouter API key (fallback)
  REDIS_URL                   — Redis connection URL (default: redis://localhost:6379/0)
  QDRANT_URL                  — Qdrant HTTP endpoint (default: http://localhost:6333)
  QDRANT_API_KEY              — Qdrant API key
  HF_TOKEN                    — HuggingFace token (for embeddings)
  TAVILY_API_KEY              — Tavily web search API key
  RESEARCH_BATCH_THRESHOLD    — Min queue depth before processing (default: 3)
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
from dataclasses import dataclass, field
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
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

BATCH_THRESHOLD = int(os.getenv("RESEARCH_BATCH_THRESHOLD", "3"))

# Venice model routing per desk
# venice-uncensored has no native function calling — ReAct fallback handles it
VENICE_MODEL_UNCENSORED = os.getenv("VENICE_MODEL_UNCENSORED", "venice-uncensored")
VENICE_MODEL_CYBER = os.getenv("VENICE_MODEL_CYBER", "mistral-31-24b")
VENICE_MODEL_DEFAULT = os.getenv("VENICE_MODEL_DEFAULT", "mistral-small-3-2-24b-instruct")

# Desks that require uncensored reasoning (no guardrails)
UNCENSORED_DESKS = {
    "human-intelligence-and-profiling-desk",
    "cultural-and-theological-intelligence-desk",
    "geopolitical-and-security-desk",
}
CYBER_DESKS = {
    "cyber-intelligence-and-warfare-desk",
}

RESEARCH_COLLECTION = "osia_research_cache"
EMBEDDING_DIM = 384
CHUNK_SIZE = 400   # words
MAX_ROUNDS = 6


def _model_for_desk(desk: str) -> str:
    if desk in UNCENSORED_DESKS:
        return VENICE_MODEL_UNCENSORED
    if desk in CYBER_DESKS:
        return VENICE_MODEL_CYBER
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
        return bool(self._r.sismember("osia:research:seen_topics", key))

    def mark_seen(self, key: str):
        self._r.sadd("osia:research:seen_topics", key)


# ---------------------------------------------------------------------------
# Qdrant client
# ---------------------------------------------------------------------------


class QdrantClient:
    def __init__(self, http: httpx.AsyncClient):
        self._http = http
        self._headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}

    async def ensure_collection(self):
        check = await self._http.get(
            f"{QDRANT_URL}/collections/{RESEARCH_COLLECTION}",
            headers=self._headers,
        )
        if check.status_code == 200:
            return
        resp = await self._http.put(
            f"{QDRANT_URL}/collections/{RESEARCH_COLLECTION}",
            headers=self._headers,
            json={"vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"}},
        )
        resp.raise_for_status()
        logger.info("Created Qdrant collection: %s", RESEARCH_COLLECTION)

    async def upsert_points(self, points: list[dict]):
        resp = await self._http.put(
            f"{QDRANT_URL}/collections/{RESEARCH_COLLECTION}/points",
            headers=self._headers,
            json={"points": points},
            timeout=30.0,
        )
        resp.raise_for_status()
        logger.info("Upserted %d points into %s", len(points), RESEARCH_COLLECTION)


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


async def tool_search_web(query: str, http: httpx.AsyncClient) -> str:
    if not TAVILY_API_KEY:
        return "Tavily API key not configured."
    try:
        resp = await http.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 5},
            timeout=15.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return "\n\n".join(
            f"[{r.get('title', '')}]({r.get('url', '')})\n{r.get('content', '')[:500]}"
            for r in results
        )
    except Exception as e:
        return f"Web search error: {e}"


async def tool_search_wikipedia(query: str, http: httpx.AsyncClient) -> str:
    try:
        resp = await http.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "srlimit": 3, "format": "json", "utf8": 1},
            timeout=10.0,
        )
        resp.raise_for_status()
        hits = resp.json().get("query", {}).get("search", [])
        if not hits:
            return "No Wikipedia results found."
        title = hits[0]["title"]
        ex = await http.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "prop": "extracts", "exintro": True,
                    "explaintext": True, "titles": title, "format": "json"},
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
    try:
        resp = await http.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": 3, "fields": "title,abstract,year,authors,url"},
            timeout=15.0,
        )
        resp.raise_for_status()
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


TOOL_REGISTRY = {
    "search_web": tool_search_web,
    "search_wikipedia": tool_search_wikipedia,
    "search_arxiv": tool_search_arxiv,
    "search_semantic_scholar": tool_search_semantic_scholar,
}

# OpenAI-format tool schemas for OpenRouter
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the live web for current events and news.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for factual background.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": "Search ArXiv for academic papers.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_semantic_scholar",
            "description": "Search Semantic Scholar for peer-reviewed literature.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
]

# ReAct pattern — parsed when native tool_calls are absent
_REACT_PATTERN = re.compile(
    r"(?:SEARCH_WEB|SEARCH_WIKIPEDIA|SEARCH_ARXIV|SEARCH_SEMANTIC_SCHOLAR):\s*(.+)",
    re.IGNORECASE,
)
_REACT_TOOL_MAP = {
    "search_web": "search_web",
    "search_wikipedia": "search_wikipedia",
    "search_arxiv": "search_arxiv",
    "search_semantic_scholar": "search_semantic_scholar",
}


def _parse_react(text: str) -> list[tuple[str, str]]:
    """Extract (tool_name, query) pairs from ReAct-style model output."""
    calls = []
    for line in text.splitlines():
        for name in _REACT_TOOL_MAP:
            prefix = name.upper().replace("_", "_") + ":"
            if line.upper().startswith(prefix):
                query = line[len(prefix):].strip()
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

    messages = [
        {
            "role": "system",
            "content": (
                f"You are an OSINT research analyst for the {job.desk}. "
                "Conduct thorough multi-source research and produce a comprehensive "
                f"intelligence brief with citations.{directives}\n\n"
                "If native tool calling is unavailable, use the ReAct format:\n"
                "SEARCH_WEB: <query>\nSEARCH_WIKIPEDIA: <query>\n"
                "SEARCH_ARXIV: <query>\nSEARCH_SEMANTIC_SCHOLAR: <query>"
            ),
        },
        {"role": "user", "content": f"Research topic: {job.topic}"},
    ]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    logger.info("Researching: %s (desk: %s, model: %s)", job.topic, job.desk, model)

    for round_num in range(MAX_ROUNDS):
        payload = {
            "model": model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
            "max_tokens": 2048,
            "temperature": 0.3,
        }

        try:
            resp = await http.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
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
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
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

    tools = [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="search_web",
            description="Search the live web for current events and news.",
            parameters=types.Schema(type="OBJECT", properties={"query": types.Schema(type="STRING")}, required=["query"]),
        ),
        types.FunctionDeclaration(
            name="search_wikipedia",
            description="Search Wikipedia for factual background.",
            parameters=types.Schema(type="OBJECT", properties={"query": types.Schema(type="STRING")}, required=["query"]),
        ),
        types.FunctionDeclaration(
            name="search_arxiv",
            description="Search ArXiv for academic papers.",
            parameters=types.Schema(type="OBJECT", properties={"query": types.Schema(type="STRING")}, required=["query"]),
        ),
        types.FunctionDeclaration(
            name="search_semantic_scholar",
            description="Search Semantic Scholar for peer-reviewed literature.",
            parameters=types.Schema(type="OBJECT", properties={"query": types.Schema(type="STRING")}, required=["query"]),
        ),
    ])]

    system = (
        f"You are an OSINT research analyst for the {job.desk}. "
        "Conduct thorough research and produce a comprehensive intelligence brief with citations."
        f"{directives}"
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
            response_parts.append(types.Part(
                function_response=types.FunctionResponse(name=call.name, response={"result": result_text})
            ))
        contents.append(types.Content(role="user", parts=response_parts))

    logger.warning("Hit max rounds for topic: %s", job.topic)
    text_parts = [p.text for p in contents[-1].parts if hasattr(p, "text") and p.text]
    return "\n".join(text_parts)


async def run_research_loop(job: ResearchJob, http: httpx.AsyncClient) -> str:
    if VENICE_API_KEY:
        return await run_research_loop_openai_compat(
            job, http, base_url=VENICE_BASE_URL, api_key=VENICE_API_KEY
        )
    if OPENROUTER_API_KEY:
        logger.warning("VENICE_API_KEY not set — falling back to OpenRouter")
        return await run_research_loop_openai_compat(
            job, http, base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY,
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
    return [" ".join(words[i: i + CHUNK_SIZE]) for i in range(0, len(words), step) if words[i: i + CHUNK_SIZE]]


async def store_research(job: ResearchJob, text: str, http: httpx.AsyncClient, qdrant: QdrantClient):
    chunks = _chunk_text(text)
    if not chunks:
        logger.warning("No chunks for job %s", job.job_id)
        return
    embeddings = await embed_texts(chunks, http)
    now = datetime.now(UTC).isoformat()
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=False)):
        point_id = int(hashlib.md5(f"{job.job_id}:{i}".encode()).hexdigest()[:8], 16)  # noqa: S324
        points.append({
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
                "source": "research_worker",
            },
        })
    await qdrant.upsert_points(points)


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
        try:
            await qdrant.ensure_collection()
        except Exception as e:
            logger.error("Failed to ensure Qdrant collection: %s", e)
            return

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

    logger.info("=== Batch complete. succeeded=%d failed=%d skipped=%d ===", succeeded, failed, skipped)


if __name__ == "__main__":
    asyncio.run(main())
