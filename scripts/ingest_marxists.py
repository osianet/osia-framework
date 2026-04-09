"""
OSIA Marxists Internet Archive Ingestion

Crawls and ingests works from the Marxists Internet Archive (marxists.org)
into a 'marxists-archive' Qdrant boost collection. Uses local sentence-transformers
embeddings — no HF_TOKEN or external embedding API required.

Designed for patient, polite crawling. The default 3.5 s crawl delay is deliberately
more conservative than marxists.org's robots.txt Crawl-delay of 1 s. The site is
free, volunteer-run, and deserves gentle treatment.

Usage:
  uv run python scripts/ingest_marxists.py
  uv run python scripts/ingest_marxists.py --dry-run
  uv run python scripts/ingest_marxists.py --authors gramsci luxemburg benjamin
  uv run python scripts/ingest_marxists.py --resume
  uv run python scripts/ingest_marxists.py --enrich                    # Ollama metadata
  uv run python scripts/ingest_marxists.py --enrich --ollama-model qwen2.5:1.5b

Options:
  --dry-run             Parse and chunk but skip Qdrant writes and Redis updates
  --authors             Space-separated author slugs (default: full curated list)
  --resume              Skip authors/works already completed in Redis
  --enrich              Use local Ollama to generate entity_tags and brief summaries
  --ollama-model        Ollama model for enrichment (default: qwen2.5:1.5b)
  --ollama-url          Ollama base URL (default: http://localhost:11434)
  --limit               Cap total works ingested (useful for testing)
  --crawl-delay         Seconds between HTTP requests (default: 3.5, min: 1.0)
  --embed-batch-size    Texts per local embedding batch (default: 32)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)
  --min-text-len        Minimum chars for a chunk to be kept (default: 150)

Environment variables (from .env):
  QDRANT_URL            Qdrant URL (default: https://qdrant.osia.dev)
  QDRANT_API_KEY        Qdrant API key
  REDIS_URL             Redis URL (default: redis://localhost:6379)

No HF_TOKEN required — embeddings run locally via sentence-transformers.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urljoin, urlparse

import httpx
import redis.asyncio as aioredis
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("osia.marxists_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "marxists-archive"
EMBEDDING_DIM = 384
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SOURCE_LABEL = "Marxists Internet Archive"
BASE_URL = "https://www.marxists.org"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180

USER_AGENT = "OSIA-Framework/1.0 (open-source intelligence research; +https://osia.dev)"
DEFAULT_CRAWL_DELAY = 3.5  # seconds — intentionally more conservative than robots.txt's 1 s minimum

# Redis keys
COMPLETED_AUTHORS_KEY = "osia:marxists:completed_authors"
COMPLETED_WORKS_PREFIX = "osia:marxists:completed_works:"

# Paths disallowed by robots.txt — skip any link whose path starts with these
DISALLOWED_PREFIXES = (
    "/admin/",
    "/cgi-bin/",
    "/webstats/",
    "/chinese/update/",
    "/espanol/admin/",
    "/espanol/justo/",
    "/espanol/trotsky/ceip/",
    "/history/canada/socialisthistory/",
    "/korean/trotsky/",
    "/archive/justo/",
)

# Safety cap on index pages crawled per author to prevent runaway discovery
MAX_INDEX_PAGES_PER_AUTHOR = 300

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# Curated author catalogue
#
# slug       — used for Redis keys, entity_tags, and --authors filtering
# path       — archive path on marxists.org (handles both /archive/ and /reference/archive/)
# desk_hint  — OSIA desk that benefits most from this author's work
#
# Paths validated against the live site; 404s are skipped gracefully at runtime.
# ---------------------------------------------------------------------------

DEFAULT_AUTHORS: list[dict] = [
    # Core Marxist theory
    {"slug": "marx", "path": "/archive/marx/", "desk_hint": "geopolitical-and-security-desk"},
    {"slug": "gramsci", "path": "/archive/gramsci/", "desk_hint": "information-warfare-desk"},
    {"slug": "luxemburg", "path": "/archive/luxemburg/", "desk_hint": "geopolitical-and-security-desk"},
    {"slug": "plekhanov", "path": "/archive/plekhanov/", "desk_hint": "geopolitical-and-security-desk"},
    {"slug": "bukharin", "path": "/archive/bukharin/", "desk_hint": "geopolitical-and-security-desk"},
    # Bolshevik / revolutionary
    {"slug": "lenin", "path": "/reference/archive/lenin/", "desk_hint": "geopolitical-and-security-desk"},
    {"slug": "trotsky", "path": "/reference/archive/trotsky/", "desk_hint": "geopolitical-and-security-desk"},
    {"slug": "mao", "path": "/reference/archive/mao/", "desk_hint": "geopolitical-and-security-desk"},
    {"slug": "ho-chi-minh", "path": "/reference/archive/ho-chi-minh/", "desk_hint": "geopolitical-and-security-desk"},
    # Anti-imperialism / liberation theory
    {"slug": "guevara", "path": "/archive/guevara/", "desk_hint": "geopolitical-and-security-desk"},
    {"slug": "fanon", "path": "/reference/archive/fanon/", "desk_hint": "geopolitical-and-security-desk"},
    {"slug": "cabral", "path": "/archive/cabral/", "desk_hint": "geopolitical-and-security-desk"},
    # Ideology and information warfare theory
    {"slug": "althusser", "path": "/reference/archive/althusser/", "desk_hint": "information-warfare-desk"},
    {"slug": "debord", "path": "/reference/archive/debord/", "desk_hint": "information-warfare-desk"},
    # Frankfurt School — critical theory and media analysis
    {"slug": "adorno", "path": "/reference/archive/adorno/", "desk_hint": "information-warfare-desk"},
    {"slug": "horkheimer", "path": "/reference/archive/horkheimer/", "desk_hint": "information-warfare-desk"},
    {"slug": "marcuse", "path": "/reference/archive/marcuse/", "desk_hint": "information-warfare-desk"},
    {"slug": "benjamin", "path": "/reference/archive/benjamin/", "desk_hint": "cultural-and-theological-intelligence-desk"},
    {"slug": "fromm", "path": "/reference/archive/fromm/", "desk_hint": "cultural-and-theological-intelligence-desk"},
    # Political economy
    {"slug": "hilferding", "path": "/archive/hilferding/", "desk_hint": "finance-and-economics-directorate"},
    {"slug": "veblen", "path": "/reference/archive/veblen/", "desk_hint": "finance-and-economics-directorate"},
    # Existentialism / philosophy
    {"slug": "sartre", "path": "/reference/archive/sartre/", "desk_hint": "cultural-and-theological-intelligence-desk"},
]

# ---------------------------------------------------------------------------
# Work-type heuristics — applied to URL path substrings
# ---------------------------------------------------------------------------

_WORK_TYPE_PATTERNS: list[tuple[str, str]] = [
    ("letter", "letter"),
    ("speech", "speech"),
    ("manifesto", "manifesto"),
    ("pamphlet", "pamphlet"),
    ("notebook", "notebook"),
    ("grundrisse", "book"),
    ("thesis", "thesis"),
    ("preface", "preface"),
    ("capital", "book"),
    ("state-and-revolution", "book"),
    ("imperialism", "book"),
    ("what-is-to-be-done", "book"),
    ("prison-notebooks", "book"),
]

# Heuristic nav-text patterns — short paragraphs matching these are skipped
_NAV_RE = re.compile(
    r"^(next|previous|back|forward|return|home|index|top|contents|table of contents|"
    r"transcribed|translated|marked up|html markup|written by|copyright|©|\d+ of \d+)[\s:.]?$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware chunker with sentence-level fallback and overlap."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    def flush(parts: list[str]) -> tuple[list[str], int]:
        if parts:
            chunks.append("\n\n".join(parts))
            tail: list[str] = []
            tail_len = 0
            for part in reversed(parts):
                if tail_len + len(part) <= overlap:
                    tail.insert(0, part)
                    tail_len += len(part)
                else:
                    break
            return tail, tail_len
        return [], 0

    for para in paragraphs:
        para_len = len(para)
        if para_len > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if current_len + len(sent) > chunk_size and current_parts:
                    current_parts, current_len = flush(current_parts)
                current_parts.append(sent)
                current_len += len(sent) + 1
            continue
        if current_len + para_len > chunk_size and current_parts:
            current_parts, current_len = flush(current_parts)
        current_parts.append(para)
        current_len += para_len + 2

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return [c for c in chunks if len(c.strip()) >= 50]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    authors_seen: int = 0
    authors_skipped: int = 0
    authors_processed: int = 0
    works_seen: int = 0
    works_skipped: int = 0
    works_processed: int = 0
    chunks_produced: int = 0
    points_upserted: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> str:
        secs = int(time.time() - self.started_at)
        return f"{secs // 60}m{secs % 60:02d}s"

    def log_progress(self) -> None:
        logger.info(
            "authors=%d/%d works=%d/%d chunks=%d upserted=%d errors=%d elapsed=%s",
            self.authors_processed,
            self.authors_seen,
            self.works_processed,
            self.works_seen,
            self.chunks_produced,
            self.points_upserted,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Local embedder — sentence-transformers, no API required
# ---------------------------------------------------------------------------


class LocalEmbedder:
    """Wraps sentence-transformers for async-compatible local inference."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, batch_size: int = 32) -> None:
        self._batch_size = batch_size
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedder")
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            logger.info("Loading local embedding model '%s' …", model_name)
            self._model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded.")
        except ImportError as exc:
            raise SystemExit(
                "sentence-transformers not installed.\n"
                "Run: uv add sentence-transformers\n"
                "Then retry."
            ) from exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist(),
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Optional Ollama enricher
# ---------------------------------------------------------------------------


class OllamaEnricher:
    """Optional: use a local Ollama model to generate entity_tags and summaries."""

    def __init__(self, model: str, base_url: str, http: httpx.AsyncClient) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._http = http
        self.available = False

    async def check(self) -> bool:
        try:
            resp = await self._http.get(f"{self._base_url}/api/tags", timeout=5.0)
            self.available = resp.status_code == 200
        except Exception:
            self.available = False
        if not self.available:
            logger.warning("Ollama not reachable at %s — enrichment disabled.", self._base_url)
        else:
            logger.info("Ollama available at %s (model: %s).", self._base_url, self._model)
        return self.available

    async def _generate(self, prompt: str) -> str:
        try:
            resp = await self._http.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False},
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as exc:
            logger.debug("Ollama generate failed: %s", exc)
            return ""

    async def extract_tags(self, text: str) -> list[str]:
        prompt = (
            "Extract 6-10 entity tags from this text. Tags should be specific: "
            "names of people, organisations, historical events, political concepts, locations. "
            "Return ONLY a valid JSON array of lowercase strings, nothing else.\n\n"
            f"Text: {text[:800]}\n\nJSON array:"
        )
        output = await self._generate(prompt)
        match = re.search(r"\[.*?\]", output, re.DOTALL)
        if match:
            try:
                tags = json.loads(match.group())
                if isinstance(tags, list):
                    return [str(t).lower().strip() for t in tags if t][:10]
            except json.JSONDecodeError:
                pass
        return []

    async def summarise(self, text: str, title: str, author_slug: str) -> str:
        prompt = (
            f'Write one sentence (max 160 characters) summarising this text titled "{title}" '
            f"by {author_slug}. Be specific. Output only the sentence, no quotes.\n\n"
            f"Text: {text[:1000]}"
        )
        result = await self._generate(prompt)
        return result[:200] if result else ""


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class MIAIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.resume: bool = args.resume
        self.enrich: bool = args.enrich
        self.ollama_model: str = args.ollama_model
        self.ollama_url: str = args.ollama_url
        self.limit: int | None = args.limit
        self.crawl_delay: float = max(1.0, args.crawl_delay)  # never drop below robots.txt minimum
        self.embed_batch_size: int = args.embed_batch_size
        self.upsert_batch_size: int = args.upsert_batch_size
        self.min_text_len: int = args.min_text_len

        if args.authors:
            slug_set = set(args.authors)
            self._authors = [a for a in DEFAULT_AUTHORS if a["slug"] in slug_set]
            unknown = slug_set - {a["slug"] for a in self._authors}
            if unknown:
                logger.warning("Unknown author slugs (not in catalogue): %s", unknown)
        else:
            self._authors = list(DEFAULT_AUTHORS)

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._http: httpx.AsyncClient | None = None
        self._embedder: LocalEmbedder | None = None
        self._enricher: OllamaEnricher | None = None
        self._upsert_buffer: list[qdrant_models.PointStruct] = []
        self._total_works_ingested = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._embedder = LocalEmbedder(batch_size=self.embed_batch_size)
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        self._http = httpx.AsyncClient(
            timeout=60.0,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )

        if self.enrich:
            self._enricher = OllamaEnricher(self.ollama_model, self.ollama_url, self._http)
            await self._enricher.check()

        try:
            await self._ensure_collection()
            stats = IngestStats()

            completed_authors: set[str] = set()
            if self.resume and self._redis:
                members = await self._redis.smembers(COMPLETED_AUTHORS_KEY)
                completed_authors = set(members)
                if completed_authors:
                    logger.info("Resuming — %d authors already completed.", len(completed_authors))

            for author in self._authors:
                stats.authors_seen += 1

                if author["slug"] in completed_authors:
                    stats.authors_skipped += 1
                    logger.info("Skipping %s (already completed).", author["slug"])
                    continue

                if self.limit is not None and self._total_works_ingested >= self.limit:
                    logger.info("Work limit (%d) reached — stopping.", self.limit)
                    break

                try:
                    await self._ingest_author(author, stats)
                except Exception as exc:
                    stats.errors += 1
                    logger.warning("Error ingesting author %s: %s", author["slug"], exc, exc_info=True)
                    continue

                if not self.dry_run and self._redis:
                    await self._redis.sadd(COMPLETED_AUTHORS_KEY, author["slug"])
                stats.authors_processed += 1
                stats.log_progress()

            await self._flush_upsert_buffer(stats)
            stats.log_progress()
            logger.info("Marxists Internet Archive ingestion complete.")
        finally:
            if self._http:
                await self._http.aclose()
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()
            if self._embedder:
                self._embedder.close()

    # ------------------------------------------------------------------
    # Author ingestion
    # ------------------------------------------------------------------

    async def _ingest_author(self, author: dict, stats: IngestStats) -> None:
        slug = author["slug"]
        logger.info("Processing author: %s (%s%s)", slug, BASE_URL, author["path"])

        completed_works: set[str] = set()
        if self.resume and self._redis:
            members = await self._redis.smembers(f"{COMPLETED_WORKS_PREFIX}{slug}")
            completed_works = set(members)

        work_urls = await self._discover_work_urls(author)
        if not work_urls:
            logger.warning("No work URLs found for %s — skipping.", slug)
            return

        logger.info("Discovered %d work URLs for %s.", len(work_urls), slug)

        for work_url in work_urls:
            if self.limit is not None and self._total_works_ingested >= self.limit:
                break

            stats.works_seen += 1
            work_key = urlparse(work_url).path

            if work_key in completed_works:
                stats.works_skipped += 1
                continue

            try:
                n_chunks = await self._ingest_work(work_url, author, stats)
                if n_chunks > 0:
                    self._total_works_ingested += 1
                    stats.works_processed += 1
                    if not self.dry_run and self._redis:
                        await self._redis.sadd(f"{COMPLETED_WORKS_PREFIX}{slug}", work_key)
            except Exception as exc:
                stats.errors += 1
                logger.debug("Error ingesting %s: %s", work_url, exc)

            if len(self._upsert_buffer) >= self.upsert_batch_size:
                await self._flush_upsert_buffer(stats)

        await self._flush_upsert_buffer(stats)

    # ------------------------------------------------------------------
    # Work URL discovery — BFS from author index
    # ------------------------------------------------------------------

    async def _discover_work_urls(self, author: dict) -> list[str]:
        """BFS from the author's index page, collecting all in-subtree .htm links."""
        author_base_path = author["path"]
        start_url = f"{BASE_URL}{author_base_path}"

        all_work_urls: set[str] = set()
        index_queue: list[str] = [start_url]
        visited_indexes: set[str] = set()

        while index_queue:
            if len(visited_indexes) >= MAX_INDEX_PAGES_PER_AUTHOR:
                logger.warning(
                    "Discovery cap (%d index pages) reached for %s — stopping discovery.",
                    MAX_INDEX_PAGES_PER_AUTHOR,
                    author["slug"],
                )
                break

            url = index_queue.pop(0)
            if url in visited_indexes:
                continue
            visited_indexes.add(url)

            await asyncio.sleep(self.crawl_delay)
            html = await self._fetch_html(url)
            if not html:
                continue

            works, sub_indexes = self._extract_links(html, url, author_base_path)
            all_work_urls.update(works)

            for si in sub_indexes:
                if si not in visited_indexes:
                    index_queue.append(si)

        # The author's root index page is a navigation page, not a work — exclude it
        all_work_urls.discard(start_url)

        logger.debug(
            "Discovery for %s: %d index pages crawled, %d work URLs found.",
            author["slug"],
            len(visited_indexes),
            len(all_work_urls),
        )
        return sorted(all_work_urls)

    def _extract_links(
        self, html: str, base_url: str, author_base_path: str
    ) -> tuple[set[str], set[str]]:
        """Parse HTML and return (work_urls, sub_index_urls) for links within author's subtree."""
        soup = BeautifulSoup(html, "html.parser")
        work_urls: set[str] = set()
        sub_index_urls: set[str] = set()

        for a in soup.find_all("a", href=True):
            href = str(a["href"]).strip()
            if not href or href.startswith("#") or href.startswith("mailto:"):
                continue

            resolved = urljoin(base_url, href)
            parsed = urlparse(resolved)

            # Same domain only
            if parsed.netloc and parsed.netloc not in ("www.marxists.org", "marxists.org"):
                continue

            path = parsed.path

            # Must stay within the author's subtree
            if not path.startswith(author_base_path):
                continue

            # Respect robots.txt disallowed paths
            if any(path.startswith(prefix) for prefix in DISALLOWED_PREFIXES):
                continue

            if not (path.endswith(".htm") or path.endswith(".html")):
                continue

            clean_url = f"https://www.marxists.org{path}"
            work_urls.add(clean_url)

            # Also queue index pages for further BFS recursion
            if path.endswith("/index.htm") or path.endswith("/index.html"):
                sub_index_urls.add(clean_url)

        return work_urls, sub_index_urls

    # ------------------------------------------------------------------
    # Individual work ingestion
    # ------------------------------------------------------------------

    async def _ingest_work(self, work_url: str, author: dict, stats: IngestStats) -> int:
        """Fetch, parse, chunk and buffer a single work. Returns chunks buffered."""
        await asyncio.sleep(self.crawl_delay)
        html = await self._fetch_html(work_url)
        if not html:
            return 0

        text, meta = self._extract_work_content(html, work_url)
        if not text or len(text) < self.min_text_len:
            return 0

        chunks = chunk_text(text)
        if not chunks:
            return 0

        stats.chunks_produced += len(chunks)
        title = meta.get("title") or author["slug"]
        year: int | None = meta.get("year")

        # Use publication year for temporal decay; fall back to now for undated works
        if year and 1840 <= year <= 2025:
            ingested_at_unix = int(datetime(year, 7, 1, tzinfo=UTC).timestamp())
        else:
            ingested_at_unix = int(datetime.now(UTC).timestamp())

        # Optional Ollama enrichment runs on the first chunk as a representative sample
        entity_tags: list[str] = [author["slug"]]
        summary = ""
        if self._enricher and self._enricher.available:
            ollama_tags = await self._enricher.extract_tags(chunks[0])
            if ollama_tags:
                entity_tags = ollama_tags
                if author["slug"] not in entity_tags:
                    entity_tags.insert(0, author["slug"])
            summary = await self._enricher.summarise(chunks[0], title, author["slug"])

        base_metadata: dict = {
            "source": SOURCE_LABEL,
            "source_url": work_url,
            "author": author["slug"],
            "title": title,
            "year": year,
            "work_type": meta.get("work_type", "essay"),
            "source_publication": meta.get("source_publication", ""),
            "desk_hint": author["desk_hint"],
            "document_type": "marxist_political_text",
            "provenance": "marxists_internet_archive",
            "ingest_date": TODAY,
            "ingested_at_unix": ingested_at_unix,
            "entity_tags": entity_tags,
        }
        if summary:
            base_metadata["summary"] = summary

        for idx, chunk in enumerate(chunks):
            if len(chunk) < self.min_text_len:
                continue

            header = f"Source: {SOURCE_LABEL}\nAuthor: {author['slug']}\nTitle: {title}"
            if year:
                header += f"\nYear: {year}"
            text_with_header = f"{header}\n\n{chunk}"

            id_key = f"marxists:{author['slug']}:{urlparse(work_url).path}:chunk{idx}"
            point_id = str(uuid.UUID(bytes=hashlib.sha256(id_key.encode()).digest()[:16]))

            self._upsert_buffer.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=[0.0] * EMBEDDING_DIM,
                    payload={"text": text_with_header, "chunk_index": idx, **base_metadata},
                )
            )

        logger.debug("Buffered %d chunks for %s — %s", len(chunks), author["slug"], title[:80])
        return len(chunks)

    # ------------------------------------------------------------------
    # HTML content extraction
    # ------------------------------------------------------------------

    def _extract_work_content(self, html: str, work_url: str) -> tuple[str, dict]:
        """Parse a work page. Returns (cleaned_text, metadata_dict)."""
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
            tag.decompose()

        # Title: h3 is the MIA standard; fall back through heading hierarchy
        title = ""
        for tag_name in ("h3", "h4", "h2", "h1"):
            tag = soup.find(tag_name)
            if tag:
                candidate = tag.get_text(" ", strip=True)
                if 3 < len(candidate) < 300:
                    title = candidate
                    break
        if not title and soup.title:
            raw = soup.title.get_text(strip=True)
            title = raw.split("|")[0].split("—")[0].strip()

        # Year from URL path (four-digit segment)
        year: int | None = None
        year_match = re.search(r"/(\d{4})/", work_url)
        if year_match:
            y = int(year_match.group(1))
            if 1840 <= y <= 2025:
                year = y

        # Body: collect <p> text, filtering out short navigation snippets
        paragraphs: list[str] = []
        for p in soup.find_all("p"):
            text = p.get_text(" ", strip=True)
            if len(text) >= 40 and not _NAV_RE.match(text.strip()):
                paragraphs.append(text)

        full_text = "\n\n".join(paragraphs)

        # Source publication — "First published:" line common on MIA work pages
        source_pub = ""
        page_text = soup.get_text(" ")
        pub_match = re.search(r"First published[:\s]+([^\n\r]{10,120})", page_text, re.IGNORECASE)
        if pub_match:
            source_pub = pub_match.group(1).strip().rstrip(";.,")

        # Work type heuristic from URL keywords
        url_lower = work_url.lower()
        work_type = "essay"
        for keyword, wtype in _WORK_TYPE_PATTERNS:
            if keyword in url_lower:
                work_type = wtype
                break

        return full_text, {
            "title": title,
            "year": year,
            "work_type": work_type,
            "source_publication": source_pub,
        }

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _fetch_html(self, url: str) -> str:
        assert self._http is not None
        for attempt in range(3):
            try:
                resp = await self._http.get(url)
                if resp.status_code in (404, 410):
                    return ""
                if resp.status_code == 429:
                    wait = 60 * (attempt + 1)
                    logger.warning("Rate limited fetching %s — waiting %ds.", url, wait)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                # Reject non-HTML responses (PDF downloads etc.)
                ct = resp.headers.get("content-type", "")
                if "html" not in ct and not resp.text[:100].lstrip().startswith("<"):
                    return ""
                return resp.text
            except httpx.HTTPStatusError:
                return ""
            except Exception as exc:
                if attempt < 2:
                    await asyncio.sleep(10 * (attempt + 1))
                else:
                    logger.debug("Failed to fetch %s: %s", url, exc)
        return ""

    # ------------------------------------------------------------------
    # Embed and upsert
    # ------------------------------------------------------------------

    async def _flush_upsert_buffer(self, stats: IngestStats | None = None) -> None:
        if not self._upsert_buffer:
            return
        assert self._embedder is not None

        points = list(self._upsert_buffer)
        self._upsert_buffer.clear()

        texts = [p.payload["text"] for p in points]
        vectors = await self._embedder.embed(texts)
        for point, vector in zip(points, vectors, strict=True):
            point.vector = vector

        if not self.dry_run:
            await self._qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.debug("Upserted %d points.", len(points))

        if stats is not None:
            stats.points_upserted += len(points)

    # ------------------------------------------------------------------
    # Collection setup
    # ------------------------------------------------------------------

    async def _ensure_collection(self) -> None:
        if self.dry_run:
            logger.info("[dry-run] Skipping collection setup.")
            return
        exists = await self._qdrant.collection_exists(COLLECTION_NAME)
        if not exists:
            await self._qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(
                    size=EMBEDDING_DIM, distance=qdrant_models.Distance.COSINE
                ),
                optimizers_config=qdrant_models.OptimizersConfigDiff(indexing_threshold=1000),
            )
            logger.info("Created Qdrant collection '%s'.", COLLECTION_NAME)
        else:
            info = await self._qdrant.get_collection(COLLECTION_NAME)
            logger.info(
                "Collection '%s' already exists (%d points).",
                COLLECTION_NAME,
                info.points_count or 0,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    author_list = ", ".join(a["slug"] for a in DEFAULT_AUTHORS)
    p = argparse.ArgumentParser(
        description="Ingest Marxists Internet Archive into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="Parse and chunk but skip all writes")
    p.add_argument(
        "--authors",
        nargs="+",
        metavar="SLUG",
        help=f"Author slugs to ingest. Available: {author_list}",
    )
    p.add_argument("--resume", action="store_true", help="Skip authors/works already completed in Redis")
    p.add_argument("--enrich", action="store_true", help="Use local Ollama for entity tags and summaries")
    p.add_argument(
        "--ollama-model",
        default="qwen2.5:1.5b",
        dest="ollama_model",
        help="Ollama model for enrichment",
    )
    p.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        dest="ollama_url",
        help="Ollama API base URL",
    )
    p.add_argument("--limit", type=int, default=None, help="Maximum total works to ingest")
    p.add_argument(
        "--crawl-delay",
        type=float,
        default=DEFAULT_CRAWL_DELAY,
        dest="crawl_delay",
        help="Seconds between HTTP requests (enforced minimum: 1.0)",
    )
    p.add_argument("--embed-batch-size", type=int, default=32, dest="embed_batch_size")
    p.add_argument("--upsert-batch-size", type=int, default=64, dest="upsert_batch_size")
    p.add_argument("--min-text-len", type=int, default=150, dest="min_text_len")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logger.info(
        "Starting Marxists Internet Archive ingest | authors=%s enrich=%s dry_run=%s crawl_delay=%.1fs",
        args.authors or f"all ({len(DEFAULT_AUTHORS)})",
        args.enrich,
        args.dry_run,
        max(1.0, args.crawl_delay),
    )
    if args.dry_run:
        logger.warning("DRY RUN — no data will be written.")

    asyncio.run(MIAIngestor(args).run())


if __name__ == "__main__":
    main()
