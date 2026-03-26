"""
OSIA Research Worker — HuggingFace Spaces deployable intelligence gatherer.

Polls osia:research_queue via the Queue API, executes multi-turn research
loops using Gemini + direct HTTP tools, then writes chunked results into
Qdrant for retrieval-augmented generation at report time.

Environment variables required:
  QUEUE_API_URL          — https://queue.osia.dev
  QUEUE_API_TOKEN        — bearer token
  QUEUE_API_UA_SENTINEL  — user-agent sentinel (default: osia-worker/1)
  QDRANT_URL             — https://qdrant.osia.dev
  QDRANT_API_KEY         — qdrant api key
  GEMINI_API_KEY         — google gemini api key
  TAVILY_API_KEY         — tavily search api key
  GEMINI_MODEL_ID        — model to use (default: gemini-2.5-flash)

Run locally:
  uv run python -m src.workers.research_worker

Deploy to HuggingFace Spaces:
  See hf-spaces/research-worker/Dockerfile
"""

import asyncio
import hashlib
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("osia.research_worker")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QUEUE_API_URL = os.getenv("QUEUE_API_URL", "https://queue.osia.dev")
QUEUE_API_TOKEN = os.getenv("QUEUE_API_TOKEN", "")
QUEUE_API_UA = os.getenv("QUEUE_API_UA_SENTINEL", "osia-worker/1")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

# Qdrant collection that research results land in — separate from desk collections
RESEARCH_COLLECTION = "osia:research_cache"

# Embedding dimension for all-MiniLM-L6-v2 (matches AnythingLLM's embedder)
EMBEDDING_DIM = 384

# How many tokens per chunk when splitting research output
CHUNK_SIZE = 400  # words, approximate

# Pop timeout — how long to block waiting for a job (seconds)
POP_TIMEOUT = 20

# Max research loop rounds per job
MAX_ROUNDS = 6

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ResearchJob:
    job_id: str
    topic: str
    desk: str
    priority: str = "normal"
    sources: list[str] = field(default_factory=lambda: ["tavily", "wikipedia", "arxiv", "semantic_scholar"])
    directives_lens: bool = True
    triggered_by: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchJob":
        return cls(
            job_id=d.get("job_id", str(uuid.uuid4())),
            topic=d.get("topic", ""),
            desk=d.get("desk", "collection-directorate"),
            priority=d.get("priority", "normal"),
            sources=d.get("sources", ["tavily", "wikipedia", "arxiv", "semantic_scholar"]),
            directives_lens=d.get("directives_lens", True),
            triggered_by=d.get("triggered_by", ""),
        )


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

    async def pop(self, queue: str = "osia:research_queue", timeout: int = POP_TIMEOUT) -> dict | None:
        resp = await self._http.post(
            f"{QUEUE_API_URL}/queue/pop",
            headers=self._headers,
            json={"queue": queue, "timeout": timeout},
            timeout=timeout + 10,
        )
        resp.raise_for_status()
        return resp.json().get("payload")

    async def is_seen(self, topic_key: str) -> bool:
        resp = await self._http.post(
            f"{QUEUE_API_URL}/queue/seen/check",
            headers=self._headers,
            json={"key": "osia:research:seen_topics", "member": topic_key},
        )
        resp.raise_for_status()
        return resp.json().get("seen", False)

    async def mark_seen(self, topic_key: str):
        await self._http.post(
            f"{QUEUE_API_URL}/queue/seen/add",
            headers=self._headers,
            json={"key": "osia:research:seen_topics", "members": [topic_key]},
        )


# ---------------------------------------------------------------------------
# Qdrant client (minimal — just upsert and collection bootstrap)
# ---------------------------------------------------------------------------

class QdrantClient:
    def __init__(self, http: httpx.AsyncClient):
        self._http = http
        self._headers = {
            "api-key": QDRANT_API_KEY,
            "Content-Type": "application/json",
        }

    async def ensure_collection(self):
        """Create the research cache collection if it doesn't exist."""
        check = await self._http.get(
            f"{QDRANT_URL}/collections/{RESEARCH_COLLECTION}",
            headers=self._headers,
        )
        if check.status_code == 200:
            return
        # Create it
        resp = await self._http.put(
            f"{QDRANT_URL}/collections/{RESEARCH_COLLECTION}",
            headers=self._headers,
            json={
                "vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"},
                "optimizers_config": {"default_segment_number": 2},
            },
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
# Embedding — uses the same model as AnythingLLM (all-MiniLM-L6-v2 via HF API)
# ---------------------------------------------------------------------------

async def embed_texts(texts: list[str], http: httpx.AsyncClient) -> list[list[float]]:
    """
    Embed a batch of texts using HuggingFace Inference API (all-MiniLM-L6-v2).
    Falls back to a zero vector on failure so the worker doesn't crash.
    """
    hf_token = os.getenv("HF_TOKEN", "")
    model_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

    try:
        resp = await http.post(
            model_url,
            headers={"Authorization": f"Bearer {hf_token}"},
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Embedding failed: %s — using zero vectors", e)
        return [[0.0] * EMBEDDING_DIM for _ in texts]


# ---------------------------------------------------------------------------
# Research tools (direct HTTP, no MCP dependency)
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
        data = resp.json()
        results = data.get("results", [])
        return "\n\n".join(
            f"[{r.get('title','')}]({r.get('url','')})\n{r.get('content','')[:500]}"
            for r in results
        )
    except Exception as e:
        return f"Web search error: {e}"


async def tool_search_wikipedia(query: str, http: httpx.AsyncClient) -> str:
    try:
        resp = await http.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search", "srsearch": query,
                "srlimit": 3, "format": "json", "utf8": 1,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        hits = resp.json().get("query", {}).get("search", [])
        if not hits:
            return "No Wikipedia results found."
        # Fetch the first article extract
        title = hits[0]["title"]
        extract_resp = await http.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "prop": "extracts", "exintro": True,
                "explaintext": True, "titles": title, "format": "json",
            },
            timeout=10.0,
        )
        extract_resp.raise_for_status()
        pages = extract_resp.json().get("query", {}).get("pages", {})
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
        # Parse Atom XML minimally
        text = resp.text
        results = []
        import re
        entries = re.findall(r"<entry>(.*?)</entry>", text, re.DOTALL)
        for entry in entries[:3]:
            title = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            summary = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            link = re.search(r'<id>(.*?)</id>', entry)
            t = title.group(1).strip() if title else "Unknown"
            s = summary.group(1).strip()[:400] if summary else ""
            url = link.group(1).strip() if link else ""
            results.append(f"**{t}**\n{url}\n{s}")
        return "\n\n---\n\n".join(results) if results else "No ArXiv results found."
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
            abstract = (p.get("abstract") or "")[:400]
            results.append(
                f"**{p.get('title','')}** ({p.get('year','')})\n"
                f"Authors: {authors}\n{p.get('url','')}\n{abstract}"
            )
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Semantic Scholar error: {e}"


# Map tool names to callables
TOOL_REGISTRY = {
    "search_web": tool_search_web,
    "search_wikipedia": tool_search_wikipedia,
    "search_arxiv": tool_search_arxiv,
    "search_semantic_scholar": tool_search_semantic_scholar,
}

# Gemini function declarations for the research loop
RESEARCH_TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="search_web",
            description="Search the live web for current events, news, and real-time information.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"query": types.Schema(type="STRING", description="Search query")},
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_wikipedia",
            description="Search Wikipedia for factual background and context.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"query": types.Schema(type="STRING", description="Search term")},
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_arxiv",
            description="Search ArXiv for academic papers and technical pre-prints.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"query": types.Schema(type="STRING", description="Academic search query")},
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_semantic_scholar",
            description="Search Semantic Scholar for peer-reviewed scientific literature.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"query": types.Schema(type="STRING", description="Scientific search query")},
                required=["query"],
            ),
        ),
    ])
]


# ---------------------------------------------------------------------------
# Research loop
# ---------------------------------------------------------------------------

async def run_research_loop(job: ResearchJob, http: httpx.AsyncClient) -> str:
    """
    Multi-turn Gemini research loop. Returns the final synthesized research text.
    """
    gemini = genai.Client(api_key=GEMINI_API_KEY)

    directives_note = (
        "\n\nAnalytical lens: Apply a decolonial and socialist materialist perspective. "
        "Prioritize labor rights, anti-imperialism, ecological impact, and data sovereignty. "
        "Avoid Washington Consensus framing."
    ) if job.directives_lens else ""

    system_prompt = (
        f"You are an OSINT research analyst for the {job.desk}. "
        f"Your task is to conduct thorough research on the following topic and produce "
        f"a comprehensive intelligence brief with citations.\n\n"
        f"Topic: {job.topic}{directives_note}\n\n"
        f"Use your research tools to gather information from multiple sources. "
        f"When you have enough information, synthesize it into a structured brief."
    )

    contents = [types.Content(role="user", parts=[types.Part(text=system_prompt)])]
    config = types.GenerateContentConfig(tools=RESEARCH_TOOLS)

    for round_num in range(MAX_ROUNDS):
        response = await asyncio.to_thread(
            gemini.models.generate_content,
            model=GEMINI_MODEL,
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0]
        contents.append(candidate.content)

        function_calls = [p for p in candidate.content.parts if p.function_call]
        if not function_calls:
            # Model is done — return final text
            text_parts = [p.text for p in candidate.content.parts if p.text]
            result = "\n".join(text_parts)
            logger.info("Research loop complete after %d rounds (%d chars)", round_num + 1, len(result))
            return result

        # Execute tools
        response_parts = []
        for part in function_calls:
            call = part.function_call
            args = dict(call.args) if call.args else {}
            query = args.get("query", "")
            logger.info("Tool call: %s(%r)", call.name, query)

            tool_fn = TOOL_REGISTRY.get(call.name)
            if tool_fn:
                try:
                    result_text = await tool_fn(query, http)
                except Exception as e:
                    result_text = f"Tool error: {e}"
            else:
                result_text = f"Unknown tool: {call.name}"

            response_parts.append(
                types.Part(function_response=types.FunctionResponse(
                    name=call.name,
                    response={"result": result_text},
                ))
            )

        contents.append(types.Content(role="user", parts=response_parts))

    # Exhausted rounds — return whatever we have
    logger.warning("Research loop hit max rounds for topic: %s", job.topic)
    text_parts = [p.text for p in contents[-1].parts if hasattr(p, "text") and p.text]
    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# Chunking and storage
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_words: int = CHUNK_SIZE) -> list[str]:
    """Split text into overlapping word-count chunks."""
    words = text.split()
    chunks = []
    overlap = chunk_words // 5  # 20% overlap
    step = chunk_words - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_words])
        if chunk:
            chunks.append(chunk)
    return chunks


async def store_research(job: ResearchJob, research_text: str, http: httpx.AsyncClient, qdrant: QdrantClient):
    """Chunk, embed, and upsert research output into Qdrant."""
    chunks = _chunk_text(research_text)
    if not chunks:
        logger.warning("No chunks to store for job %s", job.job_id)
        return

    logger.info("Embedding %d chunks for job %s", len(chunks), job.job_id)
    embeddings = await embed_texts(chunks, http)

    now = datetime.now(UTC).isoformat()
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=False)):
        # Deterministic ID from job_id + chunk index — md5 is fine here (not security-sensitive)
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
    logger.info("Stored %d chunks for topic: %s", len(points), job.topic)


# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------

async def worker_loop():
    logger.info("OSIA Research Worker starting up...")
    logger.info("Queue API: %s", QUEUE_API_URL)
    logger.info("Qdrant:    %s", QDRANT_URL)
    logger.info("Model:     %s", GEMINI_MODEL)

    if not QUEUE_API_TOKEN:
        logger.error("QUEUE_API_TOKEN not set — cannot connect to queue")
        return
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set")
        return

    async with httpx.AsyncClient(timeout=60.0) as http:
        queue = QueueClient(http)
        qdrant = QdrantClient(http)

        # Ensure the Qdrant collection exists before we start processing
        try:
            await qdrant.ensure_collection()
        except Exception as e:
            logger.error("Failed to ensure Qdrant collection: %s", e)
            return

        logger.info("Worker ready. Polling osia:research_queue...")

        while True:
            try:
                payload = await queue.pop()
                if payload is None:
                    # Timeout — queue was empty, loop again
                    continue

                job = ResearchJob.from_dict(payload)
                logger.info("Received job %s: %r (desk: %s)", job.job_id, job.topic, job.desk)

                # Deduplication — skip if we've researched this topic recently (md5 is fine, not security-sensitive)
                topic_key = hashlib.md5(job.topic.lower().strip().encode()).hexdigest()  # noqa: S324
                if await queue.is_seen(topic_key):
                    logger.info("Topic already researched, skipping: %s", job.topic)
                    continue

                # Run the research loop
                t0 = time.monotonic()
                try:
                    research_text = await run_research_loop(job, http)
                except Exception as e:
                    logger.error("Research loop failed for job %s: %s", job.job_id, e)
                    continue

                elapsed = time.monotonic() - t0
                logger.info("Research complete in %.1fs for: %s", elapsed, job.topic)

                # Store in Qdrant
                try:
                    await store_research(job, research_text, http, qdrant)
                except Exception as e:
                    logger.error("Failed to store research for job %s: %s", job.job_id, e)
                    continue

                # Mark topic as seen so we don't re-research it
                await queue.mark_seen(topic_key)
                logger.info("Job %s complete.", job.job_id)

            except httpx.HTTPStatusError as e:
                logger.error("Queue API HTTP error: %s — retrying in 10s", e)
                await asyncio.sleep(10)
            except Exception as e:
                logger.exception("Unexpected worker error: %s — retrying in 5s", e)
                await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(worker_loop())
