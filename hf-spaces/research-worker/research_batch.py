"""
OSIA Research Batch Job — runs on HuggingFace Jobs infrastructure.

Triggered by the Pi-side trigger script when the research queue reaches
the batch threshold. This script:

  1. Determines which HF endpoint to use based on job desk routing
  2. Wakes the appropriate endpoint (Dolphin R1 24B or Hermes 3 70B)
  3. Drains osia:research_queue via the Queue API
  4. Runs a multi-turn tool-calling research loop per job
  5. Chunks, embeds, and writes results to Qdrant
  6. Exits cleanly (HF Jobs billing stops)

Environment (set as HF Job secrets):
  QUEUE_API_URL          — https://queue.osia.dev
  QUEUE_API_TOKEN        — bearer token
  QUEUE_API_UA_SENTINEL  — osia-worker/1
  QDRANT_URL             — https://qdrant.osia.dev
  QDRANT_API_KEY         — qdrant api key
  HF_TOKEN               — huggingface token (for endpoint management + embeddings)
  HF_NAMESPACE           — huggingface username or org
  TAVILY_API_KEY         — tavily search api key
  HF_ENDPOINT_DOLPHIN_24B — endpoint URL (set by provision script)
  HF_ENDPOINT_HERMES_70B  — endpoint URL (set by provision script)
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("osia.research_batch")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QUEUE_API_URL = os.environ["QUEUE_API_URL"]
QUEUE_API_TOKEN = os.environ["QUEUE_API_TOKEN"]
QUEUE_API_UA = os.getenv("QUEUE_API_UA_SENTINEL", "osia-worker/1")
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
HF_NAMESPACE = os.environ["HF_NAMESPACE"]
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

ENDPOINT_DOLPHIN = os.getenv("HF_ENDPOINT_DOLPHIN_24B", "")
ENDPOINT_HERMES = os.getenv("HF_ENDPOINT_HERMES_70B", "")

RESEARCH_COLLECTION = "osia_research_cache"
EMBEDDING_DIM = 384
CHUNK_SIZE = 400  # words
MAX_ROUNDS = 6
WAKE_TIMEOUT = int(os.getenv("HF_WAKE_TIMEOUT", "600"))
POLL_INTERVAL = 10

# Desk → endpoint routing
# Dolphin R1: uncensored reasoning — HUMINT, Cultural, Geopolitical
# Hermes 70B: tool-calling + permissive — Cyber, Science, Finance, default
DOLPHIN_DESKS = {
    "human-intelligence-and-profiling-desk",
    "cultural-and-theological-intelligence-desk",
    "geopolitical-and-security-desk",
}


def _endpoint_for_desk(desk: str) -> tuple[str, str]:
    """Return (endpoint_url, model_name) for a given desk slug."""
    if desk in DOLPHIN_DESKS:
        return ENDPOINT_DOLPHIN, "cognitivecomputations/Dolphin3.0-R1-Mistral-24B"
    return ENDPOINT_HERMES, "NousResearch/Hermes-3-Llama-3.1-70B"


# ---------------------------------------------------------------------------
# Endpoint wake-up (reuses logic from hf_endpoint_manager.py)
# ---------------------------------------------------------------------------


def _wake_endpoint(endpoint_name: str) -> str | None:
    """Block until the named HF endpoint is running. Returns URL or None."""
    from huggingface_hub import get_inference_endpoint
    from huggingface_hub.utils import HfHubHTTPError

    try:
        ep = get_inference_endpoint(endpoint_name, namespace=HF_NAMESPACE, token=HF_TOKEN)
    except HfHubHTTPError:
        logger.error("Endpoint '%s' not found", endpoint_name)
        return None

    logger.info("Endpoint '%s' status: %s", endpoint_name, ep.status)

    if ep.status in ("paused", "scaledToZero"):
        logger.info("Waking endpoint '%s'...", endpoint_name)
        ep.resume()
    elif ep.status == "failed":
        logger.error("Endpoint '%s' is in failed state", endpoint_name)
        return None

    deadline = time.monotonic() + WAKE_TIMEOUT
    while time.monotonic() < deadline:
        if ep.status == "running":
            logger.info("Endpoint '%s' ready at %s", endpoint_name, ep.url)
            return ep.url
        time.sleep(POLL_INTERVAL)
        try:
            ep = get_inference_endpoint(endpoint_name, namespace=HF_NAMESPACE, token=HF_TOKEN)
            logger.info("Endpoint '%s' status: %s", endpoint_name, ep.status)
        except HfHubHTTPError:
            continue

    logger.error("Endpoint '%s' did not become ready within %ds", endpoint_name, WAKE_TIMEOUT)
    return None


async def _probe_endpoint(url: str, timeout: int = 60) -> bool:
    """Hit /v1/models until the model is actually serving."""
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient(timeout=10.0) as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get(
                    f"{url}/v1/models",
                    headers={"Authorization": f"Bearer {HF_TOKEN}"},
                )
                if resp.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            await asyncio.sleep(5)
    return False


# ---------------------------------------------------------------------------
# Queue API client
# ---------------------------------------------------------------------------


class QueueClient:
    def __init__(self, http: httpx.AsyncClient):
        self._http = http
        self._headers = {
            "Authorization": f"Bearer {QUEUE_API_TOKEN}",
            "User-Agent": QUEUE_API_UA,
            "Content-Type": "application/json",
        }

    async def pop(self, timeout: int = 2) -> dict | None:
        """Non-blocking pop — timeout=2 so we drain quickly. Retries on 429."""
        for attempt in range(5):
            resp = await self._http.post(
                f"{QUEUE_API_URL}/queue/pop",
                headers=self._headers,
                json={"queue": "osia:research_queue", "timeout": timeout},
                timeout=15,
            )
            if resp.status_code == 429:
                backoff = 2 ** attempt
                logger.warning("Queue API rate-limited (429) — backing off %ds", backoff)
                await asyncio.sleep(backoff)
                continue
            resp.raise_for_status()
            return resp.json().get("payload")
        raise RuntimeError("Queue API rate-limited after 5 retries")

    async def depth(self) -> int:
        resp = await self._http.get(
            f"{QUEUE_API_URL}/queue/length",
            headers=self._headers,
            params={"queue": "osia:research_queue"},
        )
        resp.raise_for_status()
        return resp.json().get("depth", 0)

    async def is_seen(self, key: str) -> bool:
        resp = await self._http.post(
            f"{QUEUE_API_URL}/queue/seen/check",
            headers=self._headers,
            json={"key": "osia:research:seen_topics", "member": key},
        )
        resp.raise_for_status()
        return resp.json().get("seen", False)

    async def mark_seen(self, key: str):
        await self._http.post(
            f"{QUEUE_API_URL}/queue/seen/add",
            headers=self._headers,
            json={"key": "osia:research:seen_topics", "members": [key]},
        )


# ---------------------------------------------------------------------------
# Research tools (direct HTTP)
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
        return "\n\n".join(f"[{r.get('title', '')}]({r.get('url', '')})\n{r.get('content', '')[:500]}" for r in results)
    except Exception as e:
        return f"Web search error: {e}"


async def tool_search_wikipedia(query: str, http: httpx.AsyncClient) -> str:
    try:
        resp = await http.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query, "srlimit": 3, "format": "json"},
            timeout=10.0,
        )
        hits = resp.json().get("query", {}).get("search", [])
        if not hits:
            return "No Wikipedia results."
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
            timeout=10.0,
        )
        pages = ex.json().get("query", {}).get("pages", {})
        extract = next(iter(pages.values()), {}).get("extract", "")
        return f"**{title}**\n\n{extract[:2000]}"
    except Exception as e:
        return f"Wikipedia error: {e}"


async def tool_search_arxiv(query: str, http: httpx.AsyncClient) -> str:
    import re

    try:
        resp = await http.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "max_results": 3},
            timeout=15.0,
        )
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
        return "\n\n---\n\n".join(results) or "No ArXiv results."
    except Exception as e:
        return f"ArXiv error: {e}"


async def tool_search_semantic_scholar(query: str, http: httpx.AsyncClient) -> str:
    try:
        resp = await http.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": 3, "fields": "title,abstract,year,authors,url"},
            timeout=15.0,
        )
        papers = resp.json().get("data", [])
        if not papers:
            return "No Semantic Scholar results."
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


TOOLS = {
    "search_web": tool_search_web,
    "search_wikipedia": tool_search_wikipedia,
    "search_arxiv": tool_search_arxiv,
    "search_semantic_scholar": tool_search_semantic_scholar,
}

# OpenAI-format tool schemas
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


# ---------------------------------------------------------------------------
# OpenAI-compat research loop (works against Dolphin and Hermes endpoints)
# ---------------------------------------------------------------------------


async def run_research_loop(
    topic: str,
    desk: str,
    endpoint_url: str,
    model_name: str,
    http: httpx.AsyncClient,
) -> str:
    directives = (
        "\n\nAnalytical lens: Apply a decolonial and socialist materialist perspective. "
        "Prioritize labor rights, anti-imperialism, ecological impact, and data sovereignty. "
        "Avoid Washington Consensus framing."
    )

    messages = [
        {
            "role": "system",
            "content": (
                f"You are an OSINT research analyst for the {desk}. "
                f"Conduct thorough multi-source research on the given topic and produce "
                f"a comprehensive intelligence brief with citations.{directives}"
            ),
        },
        {
            "role": "user",
            "content": f"Research topic: {topic}",
        },
    ]

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    for round_num in range(MAX_ROUNDS):
        payload = {
            "model": model_name,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
            "max_tokens": 2048,
            "temperature": 0.3,
        }

        try:
            resp = await http.post(
                f"{endpoint_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # 503 = model still cold-starting, retry
            if e.response.status_code == 503:
                logger.warning("Endpoint 503 on round %d, retrying in 15s...", round_num)
                await asyncio.sleep(15)
                continue
            raise

        data = resp.json()
        choice = data["choices"][0]
        message = choice["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            # Model is done — return final text
            content = message.get("content", "")
            logger.info("Research complete after %d rounds (%d chars)", round_num + 1, len(content))
            return content

        # Execute each tool call
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = json.loads(tc["function"]["arguments"])
            query = fn_args.get("query", "")
            logger.info("Tool: %s(%r)", fn_name, query)

            tool_fn = TOOLS.get(fn_name)
            if tool_fn:
                try:
                    result = await tool_fn(query, http)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {fn_name}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
            )

    logger.warning("Hit max rounds for topic: %s", topic)
    # Return whatever the last assistant message contained
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            return m["content"]
    return ""


# ---------------------------------------------------------------------------
# Chunking, embedding, Qdrant storage
# ---------------------------------------------------------------------------


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    overlap = CHUNK_SIZE // 5
    step = CHUNK_SIZE - overlap
    return [" ".join(words[i : i + CHUNK_SIZE]) for i in range(0, len(words), step) if words[i : i + CHUNK_SIZE]]


async def _embed(texts: list[str], http: httpx.AsyncClient) -> list[list[float]]:
    try:
        resp = await http.post(
            "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Embedding failed: %s — using zero vectors", e)
        return [[0.0] * EMBEDDING_DIM for _ in texts]


async def _ensure_collection(http: httpx.AsyncClient):
    headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
    check = await http.get(f"{QDRANT_URL}/collections/{RESEARCH_COLLECTION}", headers=headers)
    if check.status_code == 200:
        return
    resp = await http.put(
        f"{QDRANT_URL}/collections/{RESEARCH_COLLECTION}",
        headers=headers,
        json={"vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"}},
    )
    resp.raise_for_status()
    logger.info("Created Qdrant collection: %s", RESEARCH_COLLECTION)


async def _store(job_id: str, topic: str, desk: str, triggered_by: str, text: str, http: httpx.AsyncClient):
    chunks = _chunk_text(text)
    if not chunks:
        return
    embeddings = await _embed(chunks, http)
    now = datetime.now(UTC).isoformat()
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=False)):
        point_id = int(hashlib.md5(f"{job_id}:{i}".encode()).hexdigest()[:8], 16)  # noqa: S324
        points.append(
            {
                "id": point_id,
                "vector": vector,
                "payload": {
                    "text": chunk,
                    "topic": topic,
                    "desk": desk,
                    "job_id": job_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "triggered_by": triggered_by,
                    "collected_at": now,
                    "source": "hf_research_batch",
                },
            }
        )
    headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
    resp = await http.put(
        f"{QDRANT_URL}/collections/{RESEARCH_COLLECTION}/points",
        headers=headers,
        json={"points": points},
        timeout=30.0,
    )
    resp.raise_for_status()
    logger.info("Stored %d chunks for: %s", len(points), topic)


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------


async def main():
    logger.info("=== OSIA Research Batch Job starting ===")
    logger.info("Queue API: %s", QUEUE_API_URL)
    logger.info("Qdrant:    %s", QDRANT_URL)

    async with httpx.AsyncClient(timeout=60.0) as http:
        queue = QueueClient(http)

        # Check queue depth first — exit early if nothing to do
        depth = await queue.depth()
        logger.info("Queue depth: %d", depth)
        if depth == 0:
            logger.info("Queue empty — nothing to do, exiting.")
            return

        await _ensure_collection(http)

        # Drain the queue, grouping jobs by endpoint to minimise cold starts.
        # 0.25s between pops keeps us well under the queue API rate limit.
        jobs: list[dict] = []
        while True:
            payload = await queue.pop(timeout=2)
            if payload is None:
                break
            jobs.append(payload)
            await asyncio.sleep(0.25)
        logger.info("Drained %d jobs from queue", len(jobs))

        if not jobs:
            return

        # Group by endpoint
        dolphin_jobs = [j for j in jobs if j.get("desk", "") in DOLPHIN_DESKS]
        hermes_jobs = [j for j in jobs if j.get("desk", "") not in DOLPHIN_DESKS]

        async def process_batch(batch: list[dict], endpoint_env: str, endpoint_name: str):
            if not batch:
                return
            endpoint_url = os.getenv(endpoint_env, "")
            if not endpoint_url:
                logger.error("%s not set — cannot process %d jobs", endpoint_env, len(batch))
                return

            logger.info("Waking endpoint %s for %d jobs...", endpoint_name, len(batch))
            url = await asyncio.to_thread(_wake_endpoint, endpoint_name)
            if not url:
                logger.error("Failed to wake %s — skipping batch", endpoint_name)
                return

            ready = await _probe_endpoint(url)
            if not ready:
                logger.error("Endpoint %s never became ready — skipping", endpoint_name)
                return

            model = (
                "cognitivecomputations/Dolphin3.0-R1-Mistral-24B"
                if "DOLPHIN" in endpoint_env
                else "NousResearch/Hermes-3-Llama-3.1-70B"
            )
            logger.info("Endpoint ready. Processing %d jobs with %s", len(batch), model)

            for job in batch:
                topic = job.get("topic", "")
                desk = job.get("desk", "collection-directorate")
                job_id = job.get("job_id", str(uuid.uuid4()))
                triggered_by = job.get("triggered_by", "")

                if not topic:
                    continue

                topic_key = hashlib.md5(topic.lower().strip().encode()).hexdigest()  # noqa: S324
                if await queue.is_seen(topic_key):
                    logger.info("Already researched, skipping: %s", topic)
                    continue

                logger.info("Researching: %s (desk: %s)", topic, desk)
                t0 = time.monotonic()
                try:
                    text = await run_research_loop(topic, desk, url, model, http)
                except Exception as e:
                    logger.error("Research failed for '%s': %s", topic, e)
                    continue

                logger.info("Research done in %.1fs (%d chars)", time.monotonic() - t0, len(text))

                try:
                    await _store(job_id, topic, desk, triggered_by, text, http)
                except Exception as e:
                    logger.error("Storage failed for '%s': %s", topic, e)
                    continue

                await queue.mark_seen(topic_key)
                logger.info("Job complete: %s", job_id)

        # Process Dolphin batch first, then Hermes
        await process_batch(dolphin_jobs, "HF_ENDPOINT_DOLPHIN_24B", "osia-dolphin-r1-24b")
        await process_batch(hermes_jobs, "HF_ENDPOINT_HERMES_70B", "osia-hermes-70b")

    logger.info("=== Batch complete. Exiting. ===")


if __name__ == "__main__":
    asyncio.run(main())
