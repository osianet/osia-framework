"""
OSIA Church Committee Reports Ingestion

Downloads and ingests the Church Committee Final Report (1975–1976) — the landmark
US Senate investigation into CIA, FBI, NSA, and IRS intelligence abuses. Six books
covering covert operations, domestic surveillance, assassination plots, and COINTELPRO.

Sources (govinfo.gov HTML editions + archive.org fallback):
  Book I  — Foreign and Military Intelligence
  Book II — Intelligence Activities and the Rights of Americans
  Book III — Supplementary Detailed Staff Reports on Intelligence Activities
             and the Rights of Americans (COINTELPRO, NSA surveillance, IRS)
  Book IV — Supplementary Detailed Staff Reports on Foreign and Military Intelligence
  Book V  — The Investigation of the Assassination of President John F. Kennedy:
             Performance of the Intelligence Agencies
  Book VI — Supplementary Reports on Intelligence Activities

Usage:
  uv run python scripts/ingest_church_committee.py
  uv run python scripts/ingest_church_committee.py --dry-run
  uv run python scripts/ingest_church_committee.py --books 1 2 3
  uv run python scripts/ingest_church_committee.py --resume
  uv run python scripts/ingest_church_committee.py --enqueue-notable

Options:
  --dry-run             Parse and chunk but skip Qdrant writes and Redis updates
  --books               Space-separated list of book numbers to ingest (default: all 1–6)
  --resume              Skip books already in Redis completed set
  --enqueue-notable     Push key sections to HUMINT/InfoWar desk research queues
  --embed-batch-size    Texts per HF embedding call (default: 48)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)
  --min-text-len        Minimum chars for a chunk to be kept (default: 120)

Environment variables (from .env):
  HF_TOKEN              HuggingFace token (required for embeddings)
  QDRANT_URL            Qdrant URL (default: https://qdrant.osia.dev)
  QDRANT_API_KEY        Qdrant API key
  REDIS_URL             Redis URL (default: redis://localhost:6379)

Rate limiting / polite scraping:
  - Sends User-Agent: OSIA-Framework/1.0 on all requests to govinfo.gov and archive.org
  - Waits REQUEST_DELAY (1.5s) between each request (govinfo fetch, fallback URL attempts,
    and every search page) to avoid overloading government and archive.org infrastructure
  - 3-attempt retry with 10s backoff on archive.org search API calls
  - Persistent httpx.AsyncClient reused for all requests (single connection pool per run)
  - Per-book Redis checkpoint (osia:church_committee:completed_books) supports --resume
    so interrupted runs do not re-fetch already processed books
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
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
import redis.asyncio as aioredis
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("osia.church_committee_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "church-committee"
EMBEDDING_DIM = 384
SOURCE_LABEL = "Church Committee Final Report (1975–1976)"

# Redis keys
COMPLETED_BOOKS_KEY = "osia:church_committee:completed_books"
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 225

# Polite scraping
USER_AGENT = "OSIA-Framework/1.0 (open-source intelligence research; +https://osia.dev)"
REQUEST_DELAY = 1.5  # seconds between requests to govinfo.gov / archive.org

HF_EMBED_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

# ---------------------------------------------------------------------------
# Book catalogue
# Each entry: govinfo package ID + archive.org fallback identifier + title
# govinfo HTML URL pattern: https://www.govinfo.gov/content/pkg/{pkg}/html/{pkg}.htm
# archive.org text URL pattern:
#   https://archive.org/download/{ia_id}/{ia_id}_djvu.txt
# ---------------------------------------------------------------------------

BOOKS: list[dict] = [
    {
        "number": 1,
        "title": "Book I: Foreign and Military Intelligence",
        "govinfo_pkg": "CDOC-94sdoc755",
        "ia_id": "churchcommitteebook1",
        "desk_hint": "geopolitical-and-security-desk",
        "date": "1976",
    },
    {
        "number": 2,
        "title": "Book II: Intelligence Activities and the Rights of Americans",
        "govinfo_pkg": "CDOC-94sdoc756",
        "ia_id": "churchcommitteebook2",
        "desk_hint": "information-warfare-desk",
        "date": "1976",
    },
    {
        "number": 3,
        "title": "Book III: Supplementary Detailed Staff Reports on Intelligence "
        "Activities and the Rights of Americans",
        "govinfo_pkg": "CDOC-94sdoc757",
        "ia_id": "churchcommitteebook3",
        "desk_hint": "information-warfare-desk",
        "date": "1976",
    },
    {
        "number": 4,
        "title": "Book IV: Supplementary Detailed Staff Reports on Foreign and Military Intelligence",
        "govinfo_pkg": "CDOC-94sdoc758",
        "ia_id": "churchcommitteebook4",
        "desk_hint": "geopolitical-and-security-desk",
        "date": "1976",
    },
    {
        "number": 5,
        "title": "Book V: The Investigation of the Assassination of President John F. Kennedy",
        "govinfo_pkg": "CDOC-94sdoc759",
        "ia_id": "churchcommitteebook5",
        "desk_hint": "human-intelligence-and-profiling-desk",
        "date": "1976",
    },
    {
        "number": 6,
        "title": "Book VI: Supplementary Reports on Intelligence Activities",
        "govinfo_pkg": "CDOC-94sdoc760",
        "ia_id": "churchcommitteebook6",
        "desk_hint": "geopolitical-and-security-desk",
        "date": "1976",
    },
]

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


def clean_html_text(html: str) -> str:
    """Strip HTML tags and normalise whitespace for plain-text chunking."""
    # Remove script and style blocks
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Decode common HTML entities
    html = (
        html.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
        .replace("&mdash;", "—")
        .replace("&ndash;", "–")
    )
    # Normalise whitespace
    html = re.sub(r" {2,}", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


def clean_djvu_text(text: str) -> str:
    """Clean up DjVu OCR text artefacts."""
    # Remove page markers
    text = re.sub(r"\x0c", "\n\n", text)  # form feeds
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # lone page numbers
    text = re.sub(r"-\n([a-z])", r"\1", text)  # hyphenated line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    books_seen: int = 0
    books_skipped: int = 0
    books_processed: int = 0
    chunks_produced: int = 0
    points_upserted: int = 0
    docs_enqueued: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> str:
        secs = int(time.time() - self.started_at)
        return f"{secs // 60}m{secs % 60:02d}s"

    def log_progress(self) -> None:
        logger.info(
            "books=%d processed=%d skipped=%d chunks=%d upserted=%d enqueued=%d errors=%d elapsed=%s",
            self.books_seen,
            self.books_processed,
            self.books_skipped,
            self.chunks_produced,
            self.points_upserted,
            self.docs_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class ChurchCommitteeIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.book_filter: set[int] | None = set(args.books) if args.books else None
        self.resume: bool = args.resume
        self.enqueue_notable: bool = args.enqueue_notable
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.min_text_len: int = args.min_text_len

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._http: httpx.AsyncClient | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        self._http = httpx.AsyncClient(
            timeout=60.0,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        try:
            await self._ensure_collection()
            stats = IngestStats()

            completed: set[str] = set()
            if self.resume and self._redis:
                members = await self._redis.smembers(COMPLETED_BOOKS_KEY)
                completed = set(members)
                if completed:
                    logger.info("Resuming — %d books already completed.", len(completed))

            books = [b for b in BOOKS if self.book_filter is None or b["number"] in self.book_filter]
            logger.info("Processing %d Church Committee books.", len(books))

            for book in books:
                stats.books_seen += 1
                book_key = f"book{book['number']}"

                if book_key in completed:
                    stats.books_skipped += 1
                    logger.info("Skipping %s (already completed).", book["title"])
                    continue

                try:
                    await self._ingest_book(book, stats)
                except Exception as exc:
                    stats.errors += 1
                    logger.warning("Error ingesting %s: %s", book["title"], exc)
                    continue

                if not self.dry_run and self._redis:
                    await self._redis.sadd(COMPLETED_BOOKS_KEY, book_key)
                stats.books_processed += 1
                stats.log_progress()

            await self._flush_upsert_buffer(stats)
            stats.log_progress()
            logger.info("Church Committee ingestion complete.")
        finally:
            if self._http:
                await self._http.aclose()
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()

    # ------------------------------------------------------------------
    # Book fetching — try govinfo HTML first, fall back to archive.org djvu text
    # ------------------------------------------------------------------

    async def _ingest_book(self, book: dict, stats: IngestStats) -> None:
        logger.info("Fetching %s …", book["title"])
        text = await self._fetch_book_text(book)
        if not text or len(text) < 500:
            logger.warning("Could not fetch text for %s — skipping.", book["title"])
            stats.errors += 1
            return

        logger.info("Fetched %d chars for %s — chunking …", len(text), book["title"])
        chunks = chunk_text(text)
        if not chunks:
            return

        stats.chunks_produced += len(chunks)
        logger.info("Produced %d chunks for %s.", len(chunks), book["title"])

        base_metadata: dict = {
            "source": SOURCE_LABEL,
            "document_type": "congressional_report",
            "provenance": "church_committee_1975_1976",
            "ingest_date": TODAY,
            "book_number": book["number"],
            "book_title": book["title"],
            "date": book["date"],
            # approx unix timestamp for 1976-01-01 for temporal decay
            "ingested_at_unix": int(datetime(1976, 1, 1, tzinfo=UTC).timestamp()),
        }

        for idx, chunk in enumerate(chunks):
            if len(chunk) < self.min_text_len:
                continue

            header = f"Source: {SOURCE_LABEL}\nBook: {book['title']}\nDate: {book['date']}"
            text_with_header = f"{header}\n\n{chunk}"

            id_key = f"church:{book['number']}:chunk{idx}"
            point_id = str(uuid.UUID(bytes=hashlib.sha256(id_key.encode()).digest()[:16]))

            self._upsert_buffer.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=[0.0] * EMBEDDING_DIM,
                    payload={"text": text_with_header, "chunk_index": idx, **base_metadata},
                )
            )

            if len(self._upsert_buffer) >= self.upsert_batch_size:
                await self._flush_upsert_buffer(stats)

        await self._flush_upsert_buffer(stats)

        if self.enqueue_notable:
            await self._enqueue_book(book, stats)

    async def _fetch_book_text(self, book: dict) -> str:
        """Try govinfo HTML → archive.org djvu text → archive.org full text."""
        # 1. govinfo HTML
        govinfo_url = f"https://www.govinfo.gov/content/pkg/{book['govinfo_pkg']}/html/{book['govinfo_pkg']}.htm"
        html = await self._fetch_url(govinfo_url)
        if html:
            text = clean_html_text(html)
            if len(text) > 1000:
                logger.debug("Got govinfo HTML for %s.", book["title"])
                return text

        # 2. archive.org djvu text
        await asyncio.sleep(REQUEST_DELAY)
        ia_djvu_url = f"https://archive.org/download/{book['ia_id']}/{book['ia_id']}_djvu.txt"
        raw = await self._fetch_url(ia_djvu_url)
        if raw:
            text = clean_djvu_text(raw)
            if len(text) > 1000:
                logger.debug("Got archive.org djvu text for %s.", book["title"])
                return text

        # 3. archive.org full text
        await asyncio.sleep(REQUEST_DELAY)
        ia_full_url = f"https://archive.org/download/{book['ia_id']}/{book['ia_id']}_full.txt"
        raw = await self._fetch_url(ia_full_url)
        if raw:
            text = clean_djvu_text(raw)
            if len(text) > 1000:
                logger.debug("Got archive.org full text for %s.", book["title"])
                return text

        # 4. Try archive.org search to find a matching identifier
        logger.info("Trying archive.org search for '%s' …", book["title"])
        await asyncio.sleep(REQUEST_DELAY)
        alt_id = await self._search_archive_org(book["title"])
        if alt_id and alt_id != book["ia_id"]:
            for suffix in ("_djvu.txt", "_full.txt"):
                await asyncio.sleep(REQUEST_DELAY)
                url = f"https://archive.org/download/{alt_id}/{alt_id}{suffix}"
                raw = await self._fetch_url(url)
                if raw:
                    text = clean_djvu_text(raw)
                    if len(text) > 1000:
                        logger.info("Found via search — identifier: %s", alt_id)
                        return text

        return ""

    async def _search_archive_org(self, query: str) -> str | None:
        """Search archive.org for a text item matching the query, return first identifier."""
        assert self._http is not None
        search_url = "https://archive.org/advancedsearch.php"
        params = {
            "q": f'({query}) AND mediatype:texts AND subject:"government documents"',
            "fl": "identifier,title",
            "rows": "5",
            "output": "json",
        }
        for attempt in range(3):
            try:
                resp = await self._http.get(search_url, params=params)
                resp.raise_for_status()
                docs = resp.json().get("response", {}).get("docs", [])
                if docs:
                    return docs[0]["identifier"]
                return None
            except Exception as exc:
                wait = 10 * (attempt + 1)
                logger.debug("archive.org search attempt %d failed: %s — retry in %ds", attempt + 1, exc, wait)
                await asyncio.sleep(wait)
        return None

    async def _fetch_url(self, url: str) -> str:
        assert self._http is not None
        for attempt in range(3):
            try:
                resp = await self._http.get(url)
                if resp.status_code == 404:
                    return ""
                resp.raise_for_status()
                return resp.text
            except Exception as exc:
                if attempt < 2:
                    await asyncio.sleep(10 * (attempt + 1))
                else:
                    logger.debug("Failed to fetch %s: %s", url, exc)
        return ""

    # ------------------------------------------------------------------
    # Research queue
    # ------------------------------------------------------------------

    async def _enqueue_book(self, book: dict, stats: IngestStats) -> None:
        if not self._redis or self.dry_run:
            return

        redis_key = f"osia:church_committee:enqueued:book{book['number']}"
        if await self._redis.exists(redis_key):
            return

        job = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": book["title"],
                "desk": book["desk_hint"],
                "priority": "low",
                "directives_lens": True,
                "triggered_by": "church_committee_ingest",
                "metadata": {"book_number": book["number"], "source": SOURCE_LABEL},
            }
        )
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        await self._redis.set(redis_key, "1", ex=60 * 60 * 24 * 30)
        stats.docs_enqueued += 1
        logger.debug("Research job enqueued: %r → %s", book["title"], book["desk_hint"])

    # ------------------------------------------------------------------
    # Flush + embed
    # ------------------------------------------------------------------

    async def _flush_upsert_buffer(self, stats: IngestStats | None = None) -> None:
        if not self._upsert_buffer:
            return
        points = list(self._upsert_buffer)
        self._upsert_buffer.clear()
        texts = [p.payload["text"] for p in points]
        vectors = await self._embed_all(texts)
        for point, vector in zip(points, vectors, strict=True):
            point.vector = vector
        if not self.dry_run:
            await self._qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.debug("Upserted %d points.", len(points))
        if stats is not None:
            stats.points_upserted += len(points)

    async def _embed_all(self, texts: list[str]) -> list[list[float]]:
        batches = [texts[i : i + self.embed_batch_size] for i in range(0, len(texts), self.embed_batch_size)]
        results: list[list[float]] = []
        for i in range(0, len(batches), self.embed_concurrency):
            group = batches[i : i + self.embed_concurrency]
            group_results = await asyncio.gather(*[self._embed_batch(b) for b in group])
            for vecs in group_results:
                results.extend(vecs)
        return results

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with self._embed_semaphore:
            for attempt in range(4):
                try:
                    async with httpx.AsyncClient(timeout=45.0) as hf_http:
                        resp = await hf_http.post(
                            HF_EMBED_URL,
                            headers={"Authorization": f"Bearer {HF_TOKEN}"},
                            json={"inputs": texts, "options": {"wait_for_model": True}},
                        )
                        if resp.status_code == 429:
                            await asyncio.sleep(30 * (attempt + 1))
                            continue
                        resp.raise_for_status()
                        result = resp.json()
                        if isinstance(result, list) and result and isinstance(result[0], list):
                            return result
                        if isinstance(result, list) and result and isinstance(result[0], (int, float)):
                            return [result]
                        break
                except Exception as exc:
                    logger.warning("Embed attempt %d failed: %s", attempt + 1, exc)
                    await asyncio.sleep(5 * (attempt + 1))
        return [[0.0] * EMBEDDING_DIM for _ in texts]

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
                vectors_config=qdrant_models.VectorParams(size=EMBEDDING_DIM, distance=qdrant_models.Distance.COSINE),
                optimizers_config=qdrant_models.OptimizersConfigDiff(indexing_threshold=1000),
            )
            logger.info("Created Qdrant collection '%s'.", COLLECTION_NAME)
        else:
            info = await self._qdrant.get_collection(COLLECTION_NAME)
            logger.info("Collection '%s' ready (%d points).", COLLECTION_NAME, info.points_count or 0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest Church Committee Final Report into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--books", nargs="+", type=int, metavar="N", help="Book numbers to ingest (default: all 1–6)")
    p.add_argument("--resume", action="store_true", help="Skip books already in Redis completed set")
    p.add_argument(
        "--enqueue-notable",
        action="store_true",
        dest="enqueue_notable",
        help="Push each book as a research job to the relevant desk queue",
    )
    p.add_argument("--embed-batch-size", type=int, default=48, dest="embed_batch_size")
    p.add_argument("--embed-concurrency", type=int, default=3, dest="embed_concurrency")
    p.add_argument("--upsert-batch-size", type=int, default=64, dest="upsert_batch_size")
    p.add_argument("--min-text-len", type=int, default=120, dest="min_text_len")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not HF_TOKEN:
        parser.error("HF_TOKEN not set — required for embeddings.")

    logger.info(
        "Starting Church Committee ingest | books=%s enqueue_notable=%s dry_run=%s",
        args.books or "all",
        args.enqueue_notable,
        args.dry_run,
    )
    if args.dry_run:
        logger.warning("DRY RUN — no data will be written.")

    ingestor = ChurchCommitteeIngestor(args)
    asyncio.run(ingestor.run())


if __name__ == "__main__":
    main()
