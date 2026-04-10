"""
OSIA ACLED Armed Conflict Location & Event Data Ingestion

Fetches armed conflict events from the ACLED API into the 'acled-conflict-events'
Qdrant collection for Geopolitical & Security desk RAG retrieval.

ACLED covers: battles, explosions/remote violence, violence against civilians,
protests, riots, and strategic developments (coups, arrests, etc.) — with
actor names, fatality counts, narratives, and geospatial precision.

Source: https://acleddata.com/
Free API with registration. Requires ACLED_API_KEY + ACLED_EMAIL.
~900,000+ events globally since 1997.

Usage:
  uv run python scripts/ingest_acled.py
  uv run python scripts/ingest_acled.py --dry-run
  uv run python scripts/ingest_acled.py --resume
  uv run python scripts/ingest_acled.py --days-back 90
  uv run python scripts/ingest_acled.py --enqueue-notable
  uv run python scripts/ingest_acled.py --countries "Syria" "Ukraine" "Sudan"

Options:
  --dry-run             Parse and embed but skip Qdrant writes and Redis updates
  --resume              Resume from last Redis checkpoint (last processed date)
  --days-back N         Days of history to fetch on first run (default: 365)
  --countries           Only ingest events for these countries (default: all)
  --enqueue-notable     Push high-fatality events to Geopolitical desk research queue
  --limit N             Stop after N events (0 = no limit)
  --embed-batch-size    Texts per HF embedding call (default: 32)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)

Environment variables (from .env):
  ACLED_API_KEY         ACLED API key (from account settings)
  ACLED_EMAIL           ACLED registered email (required)
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
from datetime import UTC, datetime, timedelta

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
logger = logging.getLogger("osia.acled_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ACLED_API_KEY = os.getenv("ACLED_API_KEY", "")
ACLED_EMAIL = os.getenv("ACLED_EMAIL", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "acled-conflict-events"
EMBEDDING_DIM = 384
SOURCE_LABEL = "ACLED Armed Conflict Location & Event Data"

ACLED_API = "https://acleddata.com/api/acled/read"
USER_AGENT = "OSIA-Framework/1.0 (open-source intelligence research; +https://osia.dev)"
PAGE_SIZE = 500
REQUEST_DELAY = 1.5

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

CHECKPOINT_KEY = "osia:acled:last_date"
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# Event types that warrant automatic research enqueuing
NOTABLE_EVENT_TYPES = {
    "Battles",
    "Explosions/Remote violence",
    "Violence against civilians",
    "Strategic developments",
}
NOTABLE_FATALITY_THRESHOLD = 10

CHUNK_SIZE = 500
CHUNK_OVERLAP_WORDS = 60


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
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
    return [c for c in chunks if len(c.strip()) >= 60]


def _parse_date_unix(date_str: str) -> int | None:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
            return int(dt.replace(tzinfo=UTC).timestamp())
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


def build_document(event: dict) -> tuple[str, int | None]:
    """Build a narrative text document from an ACLED event record."""
    event_id = event.get("event_id_cnty", "")
    event_date = event.get("event_date", "")
    event_type = event.get("event_type", "")
    sub_event_type = event.get("sub_event_type", "")
    disorder_type = event.get("disorder_type", "")
    actor1 = event.get("actor1", "")
    assoc_actor_1 = event.get("assoc_actor_1", "")
    actor2 = event.get("actor2", "")
    assoc_actor_2 = event.get("assoc_actor_2", "")
    country = event.get("country", "")
    admin1 = event.get("admin1", "")
    admin2 = event.get("admin2", "")
    location = event.get("location", "")
    fatalities = event.get("fatalities", "0")
    notes = event.get("notes", "")
    source = event.get("source", "")
    source_scale = event.get("source_scale", "")
    latitude = event.get("latitude", "")
    longitude = event.get("longitude", "")
    tags = event.get("tags", "")

    event_unix = _parse_date_unix(event_date)

    lines: list[str] = []
    if event_type:
        lines.append(f"Event Type: {event_type}" + (f" — {sub_event_type}" if sub_event_type else ""))
    if disorder_type:
        lines.append(f"Disorder Type: {disorder_type}")
    if event_date:
        lines.append(f"Date: {event_date}")
    if event_id:
        lines.append(f"Event ID: {event_id}")

    location_parts = [p for p in [location, admin2, admin1, country] if p]
    if location_parts:
        lines.append(f"Location: {', '.join(location_parts)}")
    if latitude and longitude:
        lines.append(f"Coordinates: {latitude}, {longitude}")

    if actor1:
        actor1_str = actor1
        if assoc_actor_1:
            actor1_str += f" (with {assoc_actor_1})"
        lines.append(f"Actor 1: {actor1_str}")
    if actor2:
        actor2_str = actor2
        if assoc_actor_2:
            actor2_str += f" (with {assoc_actor_2})"
        lines.append(f"Actor 2: {actor2_str}")

    try:
        fat = int(float(fatalities))
        if fat > 0:
            lines.append(f"Fatalities: {fat}")
    except (ValueError, TypeError):
        pass

    if source:
        lines.append(f"Source: {source}" + (f" ({source_scale})" if source_scale else ""))
    if tags:
        lines.append(f"Tags: {tags}")

    if notes:
        lines.append(f"\nNarrative:\n{notes}")

    if len(lines) < 3:
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


class AcledIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.enqueue_notable: bool = args.enqueue_notable
        self.limit: int = args.limit
        self.days_back: int = args.days_back
        self.countries: list[str] = args.countries or []
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.resume: bool = args.resume

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    async def run(self) -> None:
        if not ACLED_API_KEY or not ACLED_EMAIL:
            logger.error("ACLED_API_KEY and ACLED_EMAIL must be set in .env")
            return

        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await self._ensure_collection()

            date_from = await self._resolve_date_from()
            logger.info("Fetching ACLED events from %s onwards", date_from)

            stats = IngestStats()
            await self._ingest(stats, date_from)
            await self._flush_upsert_buffer(stats)
            stats.log_progress()
            logger.info("ACLED ingestion complete.")
        finally:
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()

    async def _resolve_date_from(self) -> str:
        if self.resume and self._redis:
            checkpoint = await self._redis.get(CHECKPOINT_KEY)
            if checkpoint:
                logger.info("Resuming from checkpoint: %s", checkpoint)
                return checkpoint
        default_date = (datetime.now(UTC) - timedelta(days=self.days_back)).strftime("%Y-%m-%d")
        return default_date

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

    async def _ingest(self, stats: IngestStats, date_from: str) -> None:
        page = 1
        latest_date_seen = date_from

        async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=60.0) as http:
            while True:
                params: dict = {
                    "key": ACLED_API_KEY,
                    "email": ACLED_EMAIL,
                    "terms": "accept",
                    "limit": PAGE_SIZE,
                    "page": page,
                    "event_date": f"{date_from}:9999-12-31",
                    "event_date_where": "BETWEEN",
                    "order": "ASC",
                    "orderby": "event_date",
                }
                if self.countries:
                    params["country"] = "|".join(self.countries)

                for attempt in range(4):
                    try:
                        resp = await http.get(ACLED_API, params=params)
                        if resp.status_code == 429:
                            wait = 35 * (attempt + 1)
                            logger.warning("ACLED 429 — waiting %ds", wait)
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()
                        data = resp.json()
                        break
                    except Exception as exc:
                        logger.warning("ACLED fetch attempt %d failed (page=%d): %s", attempt + 1, page, exc)
                        await asyncio.sleep(10 * (attempt + 1))
                else:
                    logger.error("Giving up at page %d after 4 failures.", page)
                    break

                events = data.get("data", [])
                if not events:
                    logger.info("No more events at page %d — done.", page)
                    break

                count = data.get("count", len(events))
                logger.info("Fetched %d events (page=%d)", count, page)

                for event in events:
                    stats.records_seen += 1
                    try:
                        event_date = event.get("event_date", "")
                        if event_date > latest_date_seen:
                            latest_date_seen = event_date
                        await self._process_event(event, stats)
                    except Exception as exc:
                        stats.errors += 1
                        logger.warning("Error processing event %s: %s", event.get("event_id_cnty"), exc)

                    if self.limit and stats.records_processed >= self.limit:
                        logger.info("Reached --limit %d — stopping.", self.limit)
                        await self._save_checkpoint(latest_date_seen)
                        return

                    if stats.records_processed % 2000 == 0 and stats.records_processed > 0:
                        stats.log_progress()

                await self._save_checkpoint(latest_date_seen)

                if len(events) < PAGE_SIZE:
                    logger.info("Last page reached (got %d < %d).", len(events), PAGE_SIZE)
                    break

                page += 1
                await asyncio.sleep(REQUEST_DELAY)

    async def _process_event(self, event: dict, stats: IngestStats) -> None:
        event_id = str(event.get("event_id_cnty", event.get("data_id", "")))
        doc, event_unix = build_document(event)
        if not doc.strip():
            stats.records_skipped += 1
            return

        chunks = chunk_text(doc)
        if not chunks:
            stats.records_skipped += 1
            return

        stats.records_processed += 1

        event_type = event.get("event_type", "")
        country = event.get("country", "")
        admin1 = event.get("admin1", "")
        actor1 = event.get("actor1", "")
        actor2 = event.get("actor2", "")
        event_date = event.get("event_date", "")

        try:
            fatalities = int(float(event.get("fatalities", "0") or "0"))
        except (ValueError, TypeError):
            fatalities = 0

        entity_tags = [t for t in [event_type, country, admin1, actor1, actor2] if t]
        ingest_unix = event_unix or int(time.time())

        for i, chunk in enumerate(chunks):
            point_id = str(uuid.UUID(bytes=hashlib.sha256(f"acled:{event_id}:{i}".encode()).digest()[:16]))
            payload: dict = {
                "text": chunk,
                "source": SOURCE_LABEL,
                "document_type": "conflict_event",
                "provenance": "acled",
                "ingest_date": TODAY,
                "event_id": event_id,
                "event_type": event_type,
                "country": country,
                "pub_date": event_date,
                "entity_tags": entity_tags,
                "ingested_at_unix": ingest_unix,
            }
            if fatalities > 0:
                payload["fatalities"] = fatalities
            if actor1:
                payload["actor1"] = actor1
            if actor2:
                payload["actor2"] = actor2
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
            fatalities >= NOTABLE_FATALITY_THRESHOLD or event_type in NOTABLE_EVENT_TYPES
        ):
            await self._maybe_enqueue(event_id, event_type, country, actor1, actor2, fatalities, event_date, stats)

    async def _maybe_enqueue(
        self,
        event_id: str,
        event_type: str,
        country: str,
        actor1: str,
        actor2: str,
        fatalities: int,
        event_date: str,
        stats: IngestStats,
    ) -> None:
        if not self._redis or self.dry_run:
            return
        redis_key = f"osia:acled:enqueued:{event_id}"
        if await self._redis.exists(redis_key):
            return
        topic = f"ACLED conflict event: {event_type} in {country} ({event_date})"
        if actor1 and actor2:
            topic += f" — {actor1} vs {actor2}"
        elif actor1:
            topic += f" — {actor1}"
        if fatalities >= NOTABLE_FATALITY_THRESHOLD:
            topic += f" — {fatalities} fatalities"
        job = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": topic,
                "desk": "geopolitical-and-security-desk",
                "priority": "normal",
                "triggered_by": "acled_ingest",
                "metadata": {
                    "event_id": event_id,
                    "event_type": event_type,
                    "country": country,
                    "fatalities": fatalities,
                },
            }
        )
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        await self._redis.set(redis_key, "1", ex=60 * 60 * 24 * 30)
        stats.events_enqueued += 1
        logger.debug("Enqueued research: %r", topic[:100])

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

    async def _save_checkpoint(self, date: str) -> None:
        if self.dry_run or not self._redis:
            return
        await self._redis.set(CHECKPOINT_KEY, date)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest ACLED conflict events into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="Skip Qdrant writes and Redis updates")
    p.add_argument("--resume", action="store_true", help="Resume from last Redis checkpoint date")
    p.add_argument("--days-back", type=int, default=365, help="Days of history to fetch on first run")
    p.add_argument("--countries", nargs="+", help="Only ingest events for these countries")
    p.add_argument("--enqueue-notable", action="store_true", help="Push high-fatality events to research queue")
    p.add_argument("--limit", type=int, default=0, help="Stop after N events (0=no limit)")
    p.add_argument("--embed-batch-size", type=int, default=32, help="Texts per HF embedding call")
    p.add_argument("--embed-concurrency", type=int, default=3, help="Parallel embedding calls")
    p.add_argument("--upsert-batch-size", type=int, default=64, help="Points per Qdrant upsert call")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    ingestor = AcledIngestor(args)
    asyncio.run(ingestor.run())
