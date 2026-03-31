"""
OSIA CIA CREST Declassified Documents Ingestion

Ingests documents from the CIA Records Search Tool (CREST) — the agency's
25-year program archive of automatically declassified records. Sourced from
the archive.org `ciaindexed` collection, which mirrors CREST with OCR text.

Coverage: Cold War era intelligence assessments, NIEs (National Intelligence
Estimates), covert operations records, HUMINT reports, and internal CIA
directives. Primarily 1940s–1990s.

Source: https://archive.org/details/ciaindexed
  - Uses archive.org Advanced Search API to enumerate items
  - Downloads OCR'd text files (_djvu.txt) per item
  - Pagination via archive.org's cursor/rows mechanism

Usage:
  uv run python scripts/ingest_cia_crest.py
  uv run python scripts/ingest_cia_crest.py --dry-run
  uv run python scripts/ingest_cia_crest.py --limit 500
  uv run python scripts/ingest_cia_crest.py --resume
  uv run python scripts/ingest_cia_crest.py --enqueue-notable
  uv run python scripts/ingest_cia_crest.py --subject-filter "national intelligence estimate"

Options:
  --dry-run             Parse and chunk but skip Qdrant writes and Redis updates
  --limit N             Stop after N archive.org items (0 = no limit)
  --resume              Resume from last Redis checkpoint (item offset)
  --enqueue-notable     Push items with strategic keywords to desk research queues
  --subject-filter      Only ingest items containing this string in title/subject
  --search-rows         Items per archive.org search page (default: 100)
  --embed-batch-size    Texts per HF embedding call (default: 48)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)
  --min-text-len        Minimum chars for a text chunk (default: 120)

Environment variables (from .env):
  HF_TOKEN              HuggingFace token (required for embeddings)
  QDRANT_URL            Qdrant URL (default: https://qdrant.osia.dev)
  QDRANT_API_KEY        Qdrant API key
  REDIS_URL             Redis URL (default: redis://localhost:6379)

Rate limiting / polite scraping:
  - Sends User-Agent: OSIA-Framework/1.0 on all requests to archive.org
  - Waits ITEM_DELAY (1.0s) before fetching each item's OCR text, so bursts of
    sequential item downloads are paced to ~1 item/second
  - Waits PAGE_DELAY (2.0s) after processing each search result page before fetching
    the next, giving archive.org's API time to recover between paginated calls
  - Persistent httpx.AsyncClient reused for all requests (single connection pool per run)
  - Integer Redis checkpoint (osia:cia_crest:checkpoint) stores last processed offset;
    --resume restarts from this offset without re-ingesting earlier items
  - Per-item deduplication via osia:cia_crest:seen:{md5} prevents re-processing if the
    same archive.org identifier appears across overlapping search pages
  - Use --subject-filter to scope a run (e.g. "national intelligence estimate") and
    --limit to cap the number of items, avoiding runaway consumption on large collections
"""

import argparse
import asyncio
import hashlib
import io
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
logger = logging.getLogger("osia.cia_crest_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "cia-crest"
EMBEDDING_DIM = 384
SOURCE_LABEL = "CIA CREST — 25-Year Program Archive (via archive.org)"

IA_SEARCH_URL = "https://archive.org/advancedsearch.php"
# archive.org hosts two views of CIA CREST:
#   collection:ciareadingroom  — 788 K metadata records, many without downloadable files
#   identifier:CIA-RDP*        — 274 K items with confirmed downloadable files (_djvu.txt + PDF)
# We use the latter so every search result is actually retrievable.
IA_SEARCH_QUERY = "identifier:CIA-RDP*"

CHECKPOINT_KEY = "osia:cia_crest:checkpoint"
SEEN_ITEMS_KEY = "osia:cia_crest:seen_items"
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 225

# Polite scraping
USER_AGENT = "OSIA-Framework/1.0 (open-source intelligence research; +https://osia.dev)"
PAGE_DELAY = 2.0  # seconds between archive.org search pages
ITEM_DELAY = 1.0  # seconds between individual item text downloads

HF_EMBED_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

# Keywords that suggest high-value documents worth enqueueing for desk research
NOTABLE_KEYWORDS = {
    "national intelligence estimate",
    "special national intelligence estimate",
    "covert action",
    "assassination",
    "counterintelligence",
    "mkultra",
    "mk ultra",
    "operation",
    "soviet",
    "nuclear",
    "iran",
    "cuba",
    "chile",
    "vietnam",
    "defector",
}


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
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


def clean_ocr_text(text: str) -> str:
    text = re.sub(r"\x0c", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"-\n([a-z])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_notable(title: str, subject: str) -> bool:
    combined = (title + " " + subject).lower()
    return any(kw in combined for kw in NOTABLE_KEYWORDS)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    items_seen: int = 0
    items_skipped: int = 0
    items_processed: int = 0
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
            "items=%d processed=%d skipped=%d chunks=%d upserted=%d enqueued=%d errors=%d elapsed=%s",
            self.items_seen,
            self.items_processed,
            self.items_skipped,
            self.chunks_produced,
            self.points_upserted,
            self.docs_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class CiaCrestIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.limit: int = args.limit
        self.resume: bool = args.resume
        self.enqueue_notable: bool = args.enqueue_notable
        self.subject_filter: str | None = args.subject_filter.lower() if args.subject_filter else None
        self.search_rows: int = args.search_rows
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.min_text_len: int = args.min_text_len

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._http: httpx.AsyncClient | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

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

            checkpoint = 0
            if self.resume and self._redis:
                val = await self._redis.get(CHECKPOINT_KEY)
                checkpoint = int(val) if val else 0
                if checkpoint:
                    logger.info("Resuming from offset %d.", checkpoint)

            await self._ingest(stats, checkpoint)

            await self._flush_upsert_buffer(stats)
            stats.log_progress()
            logger.info("CIA CREST ingestion complete.")
        finally:
            if self._http:
                await self._http.aclose()
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()

    # ------------------------------------------------------------------
    # Search + iterate archive.org items
    # ------------------------------------------------------------------

    async def _ingest(self, stats: IngestStats, start_offset: int) -> None:
        offset = start_offset
        total_fetched = 0

        while True:
            if self.limit and total_fetched >= self.limit:
                logger.info("Reached --limit %d — stopping.", self.limit)
                break

            items = await self._search_page(offset)
            if not items:
                logger.info("No more items — archive.org search exhausted.")
                break

            for item in items:
                if self.limit and total_fetched >= self.limit:
                    break

                identifier = item.get("identifier", "")
                title = item.get("title", "")
                subject = " ".join(
                    item.get("subject", []) if isinstance(item.get("subject"), list) else [item.get("subject", "")]
                )

                stats.items_seen += 1
                total_fetched += 1

                # Subject filter
                if self.subject_filter:
                    combined = (title + " " + subject).lower()
                    if self.subject_filter not in combined:
                        stats.items_skipped += 1
                        continue

                # Session dedup
                seen_key = f"osia:cia_crest:seen:{hashlib.md5(identifier.encode(), usedforsecurity=False).hexdigest()}"
                if self._redis and await self._redis.exists(seen_key):
                    stats.items_skipped += 1
                    continue

                await asyncio.sleep(ITEM_DELAY)
                try:
                    processed = await self._process_item(identifier, title, subject, item, stats)
                    if processed and self._redis and not self.dry_run:
                        await self._redis.set(seen_key, "1", ex=60 * 60 * 24 * 90)
                except Exception as exc:
                    stats.errors += 1
                    logger.warning("Error processing %s: %s", identifier, exc)

            offset += len(items)
            if not self.dry_run and self._redis:
                await self._redis.set(CHECKPOINT_KEY, offset)

            if stats.items_processed % 100 == 0:
                stats.log_progress()

            if len(items) < self.search_rows:
                break  # Last page

            await asyncio.sleep(PAGE_DELAY)

    async def _search_page(self, offset: int) -> list[dict]:
        """Fetch one page of CIA Reading Room items from archive.org."""
        assert self._http is not None
        params = {
            "q": IA_SEARCH_QUERY,
            "fl": "identifier,title,subject,description,date",
            "rows": str(self.search_rows),
            "start": str(offset),
            "output": "json",
        }
        for attempt in range(4):
            try:
                resp = await self._http.get(IA_SEARCH_URL, params=params)
                resp.raise_for_status()
                return resp.json().get("response", {}).get("docs", [])
            except Exception as exc:
                wait = 15 * (attempt + 1)
                logger.warning("Search page attempt %d failed: %s — retry in %ds", attempt + 1, exc, wait)
                await asyncio.sleep(wait)
        return []

    @staticmethod
    def _ia_download_id(identifier: str) -> str:
        """Normalise an archive.org CIA identifier to the uppercase download form.

        archive.org search returns lowercase prefixed identifiers like:
          cia-readingroom-document-cia-rdp78-04718a000300090002-9
        The actual files are stored under the uppercase CIA-RDP identifier:
          CIA-RDP78-04718A000300090002-9
        """
        prefix = "cia-readingroom-document-"
        if identifier.lower().startswith(prefix):
            return identifier[len(prefix) :].upper()
        return identifier.upper()

    # ------------------------------------------------------------------
    # Process individual item
    # ------------------------------------------------------------------

    async def _process_item(
        self,
        identifier: str,
        title: str,
        subject: str,
        item: dict,
        stats: IngestStats,
    ) -> bool:
        text = await self._fetch_item_text(identifier)
        if not text or len(text) < self.min_text_len:
            stats.items_skipped += 1
            return False

        text = clean_ocr_text(text)
        chunks = chunk_text(text)
        if not chunks:
            stats.items_skipped += 1
            return False

        stats.items_processed += 1
        stats.chunks_produced += len(chunks)

        # Approximate doc date from archive.org metadata
        doc_date = str(item.get("date", ""))
        ingested_at_unix: int | None = None
        year_m = re.search(r"\b(19[4-9]\d|20[0-2]\d)\b", doc_date)
        if year_m:
            ingested_at_unix = int(datetime(int(year_m.group(1)), 1, 1, tzinfo=UTC).timestamp())

        base_metadata: dict = {
            "source": SOURCE_LABEL,
            "document_type": "intelligence_record",
            "provenance": "cia_crest_25yr_program",
            "ingest_date": TODAY,
            "ia_identifier": identifier,
        }
        if title:
            base_metadata["doc_title"] = title
        if subject:
            base_metadata["subject"] = subject
        if doc_date:
            base_metadata["doc_date"] = doc_date
        if ingested_at_unix:
            base_metadata["ingested_at_unix"] = ingested_at_unix

        base_id = str(uuid.UUID(bytes=hashlib.sha256(f"cia:{identifier}".encode()).digest()[:16]))
        chunk_count = len(chunks)

        for idx, chunk in enumerate(chunks):
            header = f"Source: {SOURCE_LABEL}\nTitle: {title or identifier}"
            if doc_date:
                header += f"\nDate: {doc_date}"
            text_with_header = f"{header}\n\n{chunk}"

            if chunk_count == 1:
                point_id = base_id
            else:
                point_id = str(uuid.UUID(bytes=hashlib.sha256(f"cia:{identifier}:chunk{idx}".encode()).digest()[:16]))

            self._upsert_buffer.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=[0.0] * EMBEDDING_DIM,
                    payload={"text": text_with_header, "chunk_index": idx, **base_metadata},
                )
            )

            if len(self._upsert_buffer) >= self.upsert_batch_size:
                await self._flush_upsert_buffer(stats)

        if self.enqueue_notable and is_notable(title, subject):
            await self._maybe_enqueue(identifier, title, subject, stats)

        return True

    async def _fetch_item_text(self, identifier: str) -> str:
        """Fetch text for a CIA CREST item.

        Tries djvu/full text files first (using the normalised uppercase
        CIA-RDP identifier), then falls back to PDF download + extraction.
        """
        dl_id = self._ia_download_id(identifier)

        # 1. Text files (fast, no cost) — use the normalised uppercase id
        for suffix in ("_djvu.txt", "_full.txt"):
            url = f"https://archive.org/download/{dl_id}/{dl_id}{suffix}"
            raw = await self._fetch_text_url(url)
            if raw and len(raw) > 100:
                return raw

        # 2. PDF → pypdf → Claude Haiku OCR
        pdf_bytes = await self._fetch_ia_pdf(dl_id)
        if pdf_bytes:
            return await self._extract_pdf_text(pdf_bytes, identifier)

        return ""

    async def _fetch_text_url(self, url: str) -> str:
        """Fetch a text file; treats 4xx/503 as absent."""
        assert self._http is not None
        for attempt in range(3):
            try:
                resp = await self._http.get(url)
                if resp.status_code in (403, 404, 503):
                    return ""
                resp.raise_for_status()
                return resp.text
            except Exception as exc:
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    logger.debug("Failed to fetch %s: %s", url, exc)
        return ""

    async def _download_bytes(self, url: str) -> bytes:
        """Download raw bytes; treats 4xx/503 and non-PDF responses as absent."""
        assert self._http is not None
        for attempt in range(3):
            try:
                resp = await self._http.get(url)
                if resp.status_code in (403, 404, 503):
                    return b""
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "")
                if "pdf" not in ct and resp.content[:4] != b"%PDF":
                    logger.debug("Skipping non-PDF response from %s (content-type: %s)", url, ct)
                    return b""
                return resp.content
            except Exception as exc:
                if attempt < 2:
                    await asyncio.sleep(10 * (attempt + 1))
                else:
                    logger.debug("Failed to download %s: %s", url, exc)
        return b""

    async def _fetch_ia_pdf(self, dl_id: str) -> bytes:
        """Download the primary PDF from an archive.org item."""
        assert self._http is not None
        try:
            resp = await self._http.get(f"https://archive.org/metadata/{dl_id}/files")
            if resp.status_code != 200:
                return b""
            files = resp.json().get("result", [])
        except Exception as exc:
            logger.debug("Could not fetch file list for %s: %s", dl_id, exc)
            return b""

        # Prefer the original PDF over derivatives
        pdf_files = [f["name"] for f in files if str(f.get("name", "")).lower().endswith(".pdf")]
        if not pdf_files:
            return b""

        # Pick the smallest PDF (original scans are usually smallest; _text.pdf is a derivative)
        pdf_files.sort(key=lambda n: int(next((f.get("size", 0) for f in files if f.get("name") == n), 0)))
        url = f"https://archive.org/download/{dl_id}/{pdf_files[0]}"
        await asyncio.sleep(ITEM_DELAY)
        return await self._download_bytes(url)

    async def _extract_pdf_text(self, pdf_bytes: bytes, label: str = "") -> str:
        """Extract text from PDF: try pypdf first, fall back to Claude Haiku OCR."""
        if not pdf_bytes:
            return ""
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            parts = [page.extract_text() or "" for page in reader.pages]
            full_text = "\n\n".join(p for p in parts if p.strip())
            if len(full_text.strip()) > 100:
                return full_text
        except Exception as exc:
            logger.debug("pypdf failed for %s: %s", label, exc)

        logger.debug("pypdf insufficient for %s — trying Gemini Flash.", label)
        return await self._extract_pdf_with_gemini(pdf_bytes)

    async def _extract_pdf_with_gemini(self, pdf_bytes: bytes) -> str:
        """OCR a PDF with Gemini Flash. Handles multi-page splitting for large docs."""
        from google import genai
        from google.genai import types

        MAX_CHUNK_BYTES = 20 * 1024 * 1024

        async def call_gemini(chunk: bytes) -> str:
            client = genai.Client(api_key=GEMINI_API_KEY)
            for attempt in range(3):
                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[
                                types.Part.from_bytes(data=chunk, mime_type="application/pdf"),
                                "Extract all text from this document verbatim. Preserve paragraph structure. Output only the extracted text, no commentary.",
                            ],
                        ),
                    )
                    return response.text or ""
                except Exception as exc:
                    wait = 15 * (attempt + 1)
                    logger.warning("Gemini Flash OCR attempt %d failed: %s — retry in %ds", attempt + 1, exc, wait)
                    await asyncio.sleep(wait)
            return ""

        if len(pdf_bytes) <= MAX_CHUNK_BYTES:
            return await call_gemini(pdf_bytes)

        # Split into 80-page chunks
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            PAGES_PER_CHUNK = 80
            parts: list[str] = []
            for start in range(0, len(reader.pages), PAGES_PER_CHUNK):
                writer = pypdf.PdfWriter()
                for i in range(start, min(start + PAGES_PER_CHUNK, len(reader.pages))):
                    writer.add_page(reader.pages[i])
                buf = io.BytesIO()
                writer.write(buf)
                chunk_text = await call_gemini(buf.getvalue())
                if chunk_text:
                    parts.append(chunk_text)
            return "\n\n".join(parts)
        except Exception as exc:
            logger.warning("Could not split PDF: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Research queue
    # ------------------------------------------------------------------

    async def _maybe_enqueue(self, identifier: str, title: str, subject: str, stats: IngestStats) -> None:
        if not self._redis or self.dry_run:
            return
        redis_key = f"osia:cia_crest:enqueued:{identifier}"
        if await self._redis.exists(redis_key):
            return

        topic = title or f"CIA CREST document {identifier}"
        # Route: HUMINT for personnel/behaviour docs, else Geopolitical
        desk = (
            "human-intelligence-and-profiling-desk"
            if "defect" in (title + subject).lower()
            else "geopolitical-and-security-desk"
        )

        job = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": topic,
                "desk": desk,
                "priority": "low",
                "directives_lens": True,
                "triggered_by": "cia_crest_ingest",
                "metadata": {"ia_identifier": identifier, "source": SOURCE_LABEL},
            }
        )
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        await self._redis.set(redis_key, "1", ex=60 * 60 * 24 * 30)
        stats.docs_enqueued += 1

    # ------------------------------------------------------------------
    # Embed + upsert
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

    async def _ensure_collection(self) -> None:
        if self.dry_run:
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
        description="Ingest CIA CREST documents (via archive.org) into OSIA Qdrant KB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Stop after N items (0=no limit)")
    p.add_argument("--resume", action="store_true", help="Resume from Redis checkpoint offset")
    p.add_argument(
        "--enqueue-notable",
        action="store_true",
        dest="enqueue_notable",
        help="Push notable documents to desk research queues",
    )
    p.add_argument(
        "--subject-filter",
        type=str,
        default=None,
        dest="subject_filter",
        help="Only ingest items containing this string in title/subject",
    )
    p.add_argument("--search-rows", type=int, default=100, dest="search_rows", help="Items per archive.org search page")
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
        "Starting CIA CREST ingest | limit=%s subject_filter=%s enqueue=%s dry_run=%s",
        args.limit or "none",
        args.subject_filter or "none",
        args.enqueue_notable,
        args.dry_run,
    )
    if args.dry_run:
        logger.warning("DRY RUN — no data will be written.")

    ingestor = CiaCrestIngestor(args)
    asyncio.run(ingestor.run())


if __name__ == "__main__":
    main()
