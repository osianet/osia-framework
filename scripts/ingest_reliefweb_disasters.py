"""
OSIA IFRC GO Disaster Events Ingestion

Fetches humanitarian disaster and emergency event records from the IFRC GO platform
(International Federation of Red Cross and Red Crescent Societies) and ingests them
into the 'reliefweb-disasters' Qdrant collection for Environment & Ecological
Intelligence desk RAG retrieval.

IFRC GO covers: earthquakes, floods, cyclones, droughts, wildfires, epidemics, and
complex emergencies — with rich narrative summaries, field reports, affected
population data, and multi-country coverage.

Source: https://goadmin.ifrc.org/api/v2/event/
Free public API, no authentication required. ~5,900+ events.

Usage:
  uv run python scripts/ingest_reliefweb_disasters.py
  uv run python scripts/ingest_reliefweb_disasters.py --dry-run
  uv run python scripts/ingest_reliefweb_disasters.py --resume
  uv run python scripts/ingest_reliefweb_disasters.py --limit 2000
  uv run python scripts/ingest_reliefweb_disasters.py --enqueue-notable

Options:
  --dry-run             Parse and embed but skip Qdrant writes and Redis updates
  --resume              Skip pages already recorded in Redis checkpoint
  --limit N             Stop after N events (0 = no limit)
  --enqueue-notable     Push high-impact events to Environment desk research queue
  --embed-batch-size    Texts per HF embedding call (default: 32)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)

Environment variables (from .env):
  HF_TOKEN              HuggingFace token (required for embeddings)
  QDRANT_URL            Qdrant URL (default: https://qdrant.osia.dev)
  QDRANT_API_KEY        Qdrant API key
  REDIS_URL             Redis URL (default: redis://localhost:6379)
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
logger = logging.getLogger("osia.ifrc_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "reliefweb-disasters"
EMBEDDING_DIM = 384
SOURCE_LABEL = "IFRC GO Disaster Events"

IFRC_API = "https://goadmin.ifrc.org/api/v2/event/"
USER_AGENT = "OSIA-Framework/1.0 (open-source intelligence research; +https://osia.dev)"
PAGE_SIZE = 100
REQUEST_DELAY = 1.0  # seconds between pages

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

CHECKPOINT_KEY = "osia:ifrc:offset"
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# Disaster types that warrant automatic deep-research enqueuing
NOTABLE_DISASTER_TYPES = {
    "Drought",
    "Flood",
    "Heat Wave",
    "Earthquake",
    "Cyclone",
    "Wildfire",
    "Tsunami",
    "Food Insecurity",
}

# Threshold for notable: affected population
NOTABLE_AFFECTED_THRESHOLD = 100_000

CHUNK_SIZE = 600  # words
CHUNK_OVERLAP_WORDS = 80


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def _strip_html(html: str) -> str:
    """Minimal HTML tag stripper — no lxml dependency required."""
    if not html:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&rsquo;", "'", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"&#?\w+;", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Word-count aware chunker."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) >= 80]


def _parse_date_unix(date_str: str) -> int | None:
    """Parse ISO-8601 date string to Unix timestamp."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str[:26].rstrip("Z"), fmt.rstrip("Z"))
            return int(dt.replace(tzinfo=UTC).timestamp())
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


def build_document(event: dict) -> tuple[str, int | None]:
    """
    Build a rich text document from an IFRC GO event.
    Returns (document_text, disaster_date_unix_timestamp).
    """
    name = event.get("name", "")
    dtype = (event.get("dtype") or {}).get("name", "")
    summary_html = event.get("summary") or ""
    summary = _strip_html(summary_html)
    disaster_date = event.get("disaster_start_date", "")
    num_affected = event.get("num_affected")
    countries = [c.get("name", "") for c in (event.get("countries") or []) if c.get("name")]
    severity = event.get("ifrc_severity_level_display", "")
    event_id = event.get("id")
    glide = event.get("glide", "")

    event_unix = _parse_date_unix(disaster_date)

    lines: list[str] = []
    lines.append(f"Event: {name}")
    if dtype:
        lines.append(f"Disaster Type: {dtype}")
    if disaster_date:
        lines.append(f"Date: {disaster_date[:10]}")
    if countries:
        lines.append(f"Countries: {', '.join(countries)}")
    if num_affected:
        lines.append(f"Estimated Affected: {num_affected:,}")
    if severity:
        lines.append(f"IFRC Severity: {severity}")
    if glide:
        lines.append(f"GLIDE Number: {glide}")
    if event_id:
        lines.append(f"Source URL: https://go.ifrc.org/emergencies/{event_id}")

    if summary:
        lines.append("")
        lines.append(f"Summary:\n{summary}")
    elif not name:
        return "", event_unix

    return "\n".join(lines), event_unix


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    records_seen: int = 0
    records_skipped: int = 0
    records_processed: int = 0
    chunks_produced: int = 0
    points_upserted: int = 0
    events_enqueued: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> str:
        secs = int(time.time() - self.started_at)
        return f"{secs // 60}m{secs % 60:02d}s"

    def log_progress(self) -> None:
        logger.info(
            "seen=%d processed=%d skipped=%d chunks=%d upserted=%d enqueued=%d errors=%d elapsed=%s",
            self.records_seen,
            self.records_processed,
            self.records_skipped,
            self.chunks_produced,
            self.points_upserted,
            self.events_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class IFRCIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.enqueue_notable: bool = args.enqueue_notable
        self.limit: int = args.limit
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.resume: bool = args.resume

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    async def run(self) -> None:
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await self._ensure_collection()

            start_offset = 0
            if self.resume:
                start_offset = await self._load_checkpoint()
                if start_offset:
                    logger.info("Resuming from offset %d", start_offset)

            stats = IngestStats()
            await self._ingest(stats, start_offset)
            await self._flush_upsert_buffer(stats)
            stats.log_progress()
            logger.info("IFRC GO disaster events ingestion complete.")
        finally:
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()

    async def _ensure_collection(self) -> None:
        if self.dry_run:
            logger.info("[dry-run] Skipping collection creation.")
            return
        exists = await self._qdrant.collection_exists(COLLECTION_NAME)
        if not exists:
            await self._qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qdrant_models.Distance.COSINE,
                ),
                optimizers_config=qdrant_models.OptimizersConfigDiff(indexing_threshold=1000),
            )
            logger.info("Created Qdrant collection '%s'.", COLLECTION_NAME)
        else:
            info = await self._qdrant.get_collection(COLLECTION_NAME)
            logger.info("Collection '%s' ready (%d points).", COLLECTION_NAME, info.points_count or 0)

    async def _ingest(self, stats: IngestStats, start_offset: int) -> None:
        offset = start_offset

        async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=30.0) as http:
            while True:
                url = f"{IFRC_API}?format=json&limit={PAGE_SIZE}&offset={offset}&ordering=-disaster_start_date"

                for attempt in range(4):
                    try:
                        resp = await http.get(url)
                        if resp.status_code == 429:
                            wait = 30 * (attempt + 1)
                            logger.warning("IFRC 429 — waiting %ds", wait)
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()
                        data = resp.json()
                        break
                    except Exception as exc:
                        logger.warning("API fetch attempt %d failed (offset=%d): %s", attempt + 1, offset, exc)
                        await asyncio.sleep(5 * (attempt + 1))
                else:
                    logger.error("Giving up at offset %d after 4 failures.", offset)
                    break

                events = data.get("results", [])
                if not events:
                    logger.info("No more events at offset %d — done.", offset)
                    break

                total = data.get("count", 0)
                logger.info("Fetched %d events (offset=%d / total=%d)", len(events), offset, total)

                for event in events:
                    stats.records_seen += 1
                    try:
                        await self._process_event(event, stats)
                    except Exception as exc:
                        stats.errors += 1
                        logger.warning("Error processing event %s: %s", event.get("id"), exc)

                    if self.limit and stats.records_processed >= self.limit:
                        logger.info("Reached --limit %d — stopping.", self.limit)
                        await self._save_checkpoint(offset + len(events))
                        return

                    if stats.records_processed % 500 == 0 and stats.records_processed > 0:
                        stats.log_progress()

                offset += PAGE_SIZE
                await self._save_checkpoint(offset)

                if not data.get("next"):
                    logger.info("Last page reached.")
                    break

                await asyncio.sleep(REQUEST_DELAY)

    async def _process_event(self, event: dict, stats: IngestStats) -> None:
        event_id = str(event.get("id", ""))
        name = event.get("name", "")

        doc, event_unix = build_document(event)
        if not doc.strip():
            stats.records_skipped += 1
            return

        chunks = chunk_text(doc)
        if not chunks:
            stats.records_skipped += 1
            return

        stats.records_processed += 1

        dtype = (event.get("dtype") or {}).get("name", "")
        countries = [c.get("name", "") for c in (event.get("countries") or []) if c.get("name")]
        num_affected = event.get("num_affected")
        disaster_date = event.get("disaster_start_date", "")
        pub_date = disaster_date[:10] if disaster_date else ""
        severity = event.get("ifrc_severity_level_display", "")

        entity_tags = [name] + ([dtype] if dtype else []) + countries[:5]
        entity_tags = [t for t in entity_tags if t]
        ingest_unix = event_unix or int(time.time())

        for i, chunk in enumerate(chunks):
            point_id = str(uuid.UUID(bytes=hashlib.sha256(f"ifrc:{event_id}:{i}".encode()).digest()[:16]))
            payload: dict = {
                "text": chunk,
                "source": SOURCE_LABEL,
                "document_type": "disaster_event",
                "provenance": "ifrc_go",
                "ingest_date": TODAY,
                "event_id": event_id,
                "title": name,
                "pub_date": pub_date,
                "entity_tags": entity_tags,
                "ingested_at_unix": ingest_unix,
            }
            if dtype:
                payload["disaster_type"] = dtype
            if countries:
                payload["countries"] = countries
            if num_affected:
                payload["num_affected"] = num_affected
            if severity:
                payload["severity"] = severity
            if len(chunks) > 1:
                payload["chunk_index"] = i
                payload["total_chunks"] = len(chunks)

            self._upsert_buffer.append(
                qdrant_models.PointStruct(id=point_id, vector=[0.0] * EMBEDDING_DIM, payload=payload)
            )
            stats.chunks_produced += 1

        if len(self._upsert_buffer) >= self.upsert_batch_size:
            await self._flush_upsert_buffer(stats)

        if self.enqueue_notable and (
            dtype in NOTABLE_DISASTER_TYPES or (num_affected and num_affected >= NOTABLE_AFFECTED_THRESHOLD)
        ):
            await self._maybe_enqueue(event_id, name, dtype, countries, num_affected, stats)

    async def _maybe_enqueue(
        self,
        event_id: str,
        name: str,
        dtype: str,
        countries: list[str],
        num_affected: int | None,
        stats: IngestStats,
    ) -> None:
        if not self._redis or self.dry_run:
            return
        redis_key = f"osia:ifrc:enqueued:{event_id}"
        if await self._redis.exists(redis_key):
            return
        country_str = ", ".join(countries[:3]) if countries else "unknown region"
        topic = f"IFRC disaster event: {name} ({country_str})"
        if num_affected:
            topic += f" — {num_affected:,} affected"
        job = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": topic,
                "desk": "environment-and-ecology-desk",
                "priority": "normal",
                "triggered_by": "ifrc_ingest",
                "metadata": {"event_id": event_id, "disaster_type": dtype, "countries": countries},
            }
        )
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        await self._redis.set(redis_key, "1", ex=60 * 60 * 24 * 30)
        stats.events_enqueued += 1
        logger.debug("Enqueued research: %r", name[:80])

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
        for group_start in range(0, len(batches), self.embed_concurrency):
            group = batches[group_start : group_start + self.embed_concurrency]
            group_results = await asyncio.gather(*[self._embed_batch(b) for b in group])
            for batch_vecs in group_results:
                results.extend(batch_vecs)
        return results

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with self._embed_semaphore:
            for attempt in range(4):
                try:
                    async with httpx.AsyncClient(timeout=45.0) as http:
                        resp = await http.post(
                            HF_EMBEDDING_URL,
                            headers={"Authorization": f"Bearer {HF_TOKEN}"},
                            json={"inputs": texts, "options": {"wait_for_model": True}},
                        )
                        if resp.status_code == 429:
                            wait = 30 * (attempt + 1)
                            logger.warning("HF embedding 429 — waiting %ds", wait)
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()
                        result = resp.json()
                        if isinstance(result, list) and result and isinstance(result[0], list):
                            return result
                        if isinstance(result, list) and result and isinstance(result[0], (int, float)):
                            return [result]
                        logger.error("Unexpected HF embedding shape: %s", type(result))
                        break
                except Exception as exc:
                    logger.warning("Embed attempt %d failed: %s", attempt + 1, exc)
                    await asyncio.sleep(5 * (attempt + 1))
        logger.error("Embedding failed for batch of %d — using zero vectors.", len(texts))
        return [[0.0] * EMBEDDING_DIM for _ in texts]

    async def _save_checkpoint(self, offset: int) -> None:
        if self.dry_run or not self._redis:
            return
        await self._redis.set(CHECKPOINT_KEY, offset)

    async def _load_checkpoint(self) -> int:
        if not self._redis:
            return 0
        val = await self._redis.get(CHECKPOINT_KEY)
        return int(val) if val else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest IFRC GO disaster events into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="Skip Qdrant writes and Redis updates")
    p.add_argument("--resume", action="store_true", help="Resume from last Redis checkpoint")
    p.add_argument("--limit", type=int, default=0, help="Stop after N events (0=no limit)")
    p.add_argument("--enqueue-notable", action="store_true", help="Push high-impact disasters to research queue")
    p.add_argument("--embed-batch-size", type=int, default=32, help="Texts per HF embedding call")
    p.add_argument("--embed-concurrency", type=int, default=3, help="Parallel embedding calls")
    p.add_argument("--upsert-batch-size", type=int, default=64, help="Points per Qdrant upsert call")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    ingestor = IFRCIngestor(args)
    asyncio.run(ingestor.run())
