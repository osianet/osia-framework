"""
OSIA GDELT 2.0 Global Events Ingestion

Downloads and ingests events from the GDELT 2.0 project into the 'gdelt-events'
Qdrant collection for Information & Psychological Warfare desk RAG retrieval.

GDELT monitors the world's news media in 100+ languages, encoding events using
the CAMEO taxonomy. Each 15-minute file covers the globe. This script filters
for information-warfare-relevant events: high-conflict events, media-actor
events, propaganda/narrative operations, and high-coverage flashpoints.

Source: https://www.gdeltproject.org/
Free public data, no authentication required. Updated every 15 minutes.

CAMEO QuadClass:
  1 = Verbal Cooperation  2 = Material Cooperation
  3 = Verbal Conflict     4 = Material Conflict

Filter (either condition passes):
  (QuadClass >= 3 AND NumMentions >= 10)  — high-coverage conflict events
  (Actor is media type "MED" AND NumMentions >= 5)  — media-actor events

Usage:
  uv run python scripts/ingest_gdelt.py
  uv run python scripts/ingest_gdelt.py --dry-run
  uv run python scripts/ingest_gdelt.py --resume
  uv run python scripts/ingest_gdelt.py --days-back 7
  uv run python scripts/ingest_gdelt.py --enqueue-notable
  uv run python scripts/ingest_gdelt.py --min-mentions 20

Options:
  --dry-run             Parse and embed but skip Qdrant writes and Redis updates
  --resume              Resume from last Redis checkpoint (last processed file timestamp)
  --days-back N         Days to backfill on first run (default: 7)
  --min-mentions N      Minimum NumMentions to include a conflict event (default: 10)
  --enqueue-notable     Push high-coverage conflict events to InfoWar research queue
  --limit N             Stop after N events ingested (0 = no limit)
  --embed-batch-size    Texts per HF embedding call (default: 48)
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
import csv
import hashlib
import io
import json
import logging
import os
import re
import time
import uuid
import zipfile
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
logger = logging.getLogger("osia.gdelt_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "gdelt-events"
EMBEDDING_DIM = 384
SOURCE_LABEL = "GDELT 2.0 Global Event Database"

GDELT_MASTERLIST = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
USER_AGENT = "OSIA-Framework/1.0 (open-source intelligence research; +https://osia.dev)"
REQUEST_DELAY = 1.0

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

CHECKPOINT_KEY = "osia:gdelt:last_ts"
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# GDELT 2.0 export CSV column names (no header in file, 61 columns)
GDELT_COLUMNS = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat", "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat", "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]

# CAMEO event root codes relevant to information warfare
INFOWAR_ROOT_CODES = {"01", "02", "10", "11", "12", "13", "14", "17"}

# QuadClass >= 3 = verbal/material conflict
MIN_QUADCLASS_CONFLICT = 3

# Notable: high media coverage + material conflict
NOTABLE_MIN_MENTIONS = 50
NOTABLE_MIN_QUADCLASS = 4

CHUNK_SIZE = 400
CHUNK_OVERLAP_WORDS = 50


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


def _parse_file_ts(url: str) -> str | None:
    """Extract YYYYMMDDHHMMSS timestamp string from a GDELT file URL."""
    m = re.search(r"/(\d{14})\.export\.CSV\.zip", url)
    return m.group(1) if m else None


def _ts_to_unix(ts_str: str) -> int:
    """Convert GDELT timestamp string YYYYMMDDHHMMSS to Unix timestamp."""
    dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
    return int(dt.replace(tzinfo=UTC).timestamp())


def _date_to_ts_prefix(dt: datetime) -> str:
    """Return a YYYYMMDD prefix for filtering masterlist entries by date."""
    return dt.strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


def build_document(row: dict) -> tuple[str, int | None]:
    """Build a narrative document from a GDELT event row."""
    event_id = row.get("GlobalEventID", "")
    day = row.get("Day", "")
    actor1_name = row.get("Actor1Name", "")
    actor1_country = row.get("Actor1CountryCode", "")
    actor1_type = row.get("Actor1Type1Code", "")
    actor2_name = row.get("Actor2Name", "")
    actor2_country = row.get("Actor2CountryCode", "")
    actor2_type = row.get("Actor2Type1Code", "")
    event_code = row.get("EventCode", "")
    event_root = row.get("EventRootCode", "")
    quad_class = row.get("QuadClass", "")
    goldstein = row.get("GoldsteinScale", "")
    num_mentions = row.get("NumMentions", "")
    num_articles = row.get("NumArticles", "")
    avg_tone = row.get("AvgTone", "")
    action_geo = row.get("ActionGeo_FullName", "")
    action_country = row.get("ActionGeo_CountryCode", "")
    source_url = row.get("SOURCEURL", "")

    # Parse event date from Day field (YYYYMMDD)
    event_unix: int | None = None
    if day and len(day) == 8:
        try:
            dt = datetime.strptime(day, "%Y%m%d")
            event_unix = int(dt.replace(tzinfo=UTC).timestamp())
        except ValueError:
            pass

    if not (actor1_name or actor2_name or action_geo):
        return "", event_unix

    # Map QuadClass to human-readable
    quad_labels = {"1": "Verbal Cooperation", "2": "Material Cooperation",
                   "3": "Verbal Conflict", "4": "Material Conflict"}
    quad_label = quad_labels.get(str(quad_class), f"Class {quad_class}")

    date_str = f"{day[:4]}-{day[4:6]}-{day[6:8]}" if day and len(day) == 8 else day

    lines: list[str] = []
    lines.append(f"Event Type: GDELT {quad_label} (CAMEO {event_code} / root {event_root})")
    if date_str:
        lines.append(f"Date: {date_str}")

    actors = []
    if actor1_name:
        a1 = actor1_name
        if actor1_country:
            a1 += f" ({actor1_country})"
        if actor1_type:
            a1 += f" [type: {actor1_type}]"
        actors.append(f"Actor 1: {a1}")
    if actor2_name:
        a2 = actor2_name
        if actor2_country:
            a2 += f" ({actor2_country})"
        if actor2_type:
            a2 += f" [type: {actor2_type}]"
        actors.append(f"Actor 2: {a2}")
    lines.extend(actors)

    if action_geo:
        geo = action_geo
        if action_country and action_country not in action_geo:
            geo += f" ({action_country})"
        lines.append(f"Location: {geo}")

    try:
        lines.append(f"Media Coverage: {int(num_mentions)} mentions across {int(num_articles)} articles")
    except (ValueError, TypeError):
        pass

    try:
        tone_val = float(avg_tone)
        tone_desc = "hostile" if tone_val < -5 else "negative" if tone_val < 0 else "positive"
        lines.append(f"Average Tone: {tone_val:.2f} ({tone_desc})")
    except (ValueError, TypeError):
        pass

    try:
        lines.append(f"Goldstein Scale: {float(goldstein):.1f} (stability impact: -10=destabilising, +10=stabilising)")
    except (ValueError, TypeError):
        pass

    if source_url:
        lines.append(f"Source: {source_url}")
    if event_id:
        lines.append(f"GDELT Event ID: {event_id}")

    if len(lines) < 4:
        return "", event_unix

    return "\n".join(lines), event_unix


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    files_processed: int = 0
    records_seen: int = 0
    records_skipped: int = 0
    records_processed: int = 0
    points_upserted: int = 0
    events_enqueued: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> str:
        secs = int(time.time() - self.started_at)
        return f"{secs // 60}m{secs % 60:02d}s"

    def log_progress(self) -> None:
        logger.info(
            "files=%d seen=%d processed=%d skipped=%d upserted=%d enqueued=%d errors=%d elapsed=%s",
            self.files_processed,
            self.records_seen,
            self.records_processed,
            self.records_skipped,
            self.points_upserted,
            self.events_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class GdeltIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.enqueue_notable: bool = args.enqueue_notable
        self.limit: int = args.limit
        self.days_back: int = args.days_back
        self.min_mentions: int = args.min_mentions
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

            cutoff_ts = await self._resolve_cutoff_ts()
            logger.info("Processing GDELT files newer than ts=%s", cutoff_ts or "none (all)")

            stats = IngestStats()
            file_urls = await self._get_file_urls(cutoff_ts)
            logger.info("Found %d GDELT files to process", len(file_urls))

            async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=120.0) as http:
                for url, ts_str in file_urls:
                    try:
                        await self._process_file(url, ts_str, stats, http)
                        await self._save_checkpoint(ts_str)
                        stats.log_progress()
                    except Exception as exc:
                        stats.errors += 1
                        logger.error("Failed to process %s: %s", url, exc)

                    if self.limit and stats.records_processed >= self.limit:
                        logger.info("Reached --limit %d — stopping.", self.limit)
                        break

                    await asyncio.sleep(REQUEST_DELAY)

            await self._flush_upsert_buffer(stats)
            stats.log_progress()
            logger.info("GDELT ingestion complete.")
        finally:
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()

    async def _resolve_cutoff_ts(self) -> str | None:
        """Return the last processed file timestamp string, or None for full backfill."""
        if self.resume and self._redis:
            checkpoint = await self._redis.get(CHECKPOINT_KEY)
            if checkpoint:
                logger.info("Resuming from checkpoint ts=%s", checkpoint)
                return checkpoint
        cutoff_dt = datetime.now(UTC) - timedelta(days=self.days_back)
        return cutoff_dt.strftime("%Y%m%d%H%M%S")

    async def _get_file_urls(self, cutoff_ts: str | None) -> list[tuple[str, str]]:
        """Download masterfilelist.txt and return (url, ts_str) pairs newer than cutoff."""
        async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=60.0) as http:
            for attempt in range(3):
                try:
                    resp = await http.get(GDELT_MASTERLIST)
                    resp.raise_for_status()
                    break
                except Exception as exc:
                    logger.warning("Masterlist fetch attempt %d failed: %s", attempt + 1, exc)
                    await asyncio.sleep(10 * (attempt + 1))
            else:
                logger.error("Failed to fetch GDELT masterlist.")
                return []

        results: list[tuple[str, str]] = []
        for line in resp.text.splitlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            url = parts[2]
            if ".export.CSV.zip" not in url:
                continue
            ts_str = _parse_file_ts(url)
            if not ts_str:
                continue
            if cutoff_ts and ts_str <= cutoff_ts:
                continue
            results.append((url, ts_str))

        results.sort(key=lambda x: x[1])
        return results

    async def _process_file(
        self, url: str, ts_str: str, stats: IngestStats, http: httpx.AsyncClient
    ) -> None:
        logger.info("Processing GDELT file %s", ts_str)

        for attempt in range(4):
            try:
                resp = await http.get(url)
                if resp.status_code == 429:
                    await asyncio.sleep(30 * (attempt + 1))
                    continue
                resp.raise_for_status()
                zip_bytes = resp.content
                break
            except Exception as exc:
                logger.warning("Download attempt %d failed for %s: %s", attempt + 1, ts_str, exc)
                await asyncio.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"Failed to download {url}")

        rows = await asyncio.get_event_loop().run_in_executor(
            None, self._parse_zip_csv, zip_bytes
        )

        file_processed = 0
        for row in rows:
            stats.records_seen += 1
            if not self._passes_filter(row):
                stats.records_skipped += 1
                continue

            try:
                await self._process_row(row, ts_str, stats)
                file_processed += 1
            except Exception as exc:
                stats.errors += 1
                logger.debug("Row error: %s", exc)

            if self.limit and stats.records_processed >= self.limit:
                break

        await self._flush_upsert_buffer(stats)
        stats.files_processed += 1
        logger.debug("File %s: %d rows kept of %d seen", ts_str, file_processed, len(rows))

    def _parse_zip_csv(self, zip_bytes: bytes) -> list[dict]:
        """Decompress and parse a GDELT 2.0 export CSV zip."""
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = [n for n in zf.namelist() if n.endswith(".CSV")]
            if not names:
                return []
            with zf.open(names[0]) as fh:
                text = fh.read().decode("utf-8", errors="replace")
        reader = csv.DictReader(
            io.StringIO(text),
            fieldnames=GDELT_COLUMNS,
            delimiter="\t",
        )
        return list(reader)

    def _passes_filter(self, row: dict) -> bool:
        """Return True if this event is information-warfare-relevant."""
        try:
            num_mentions = int(row.get("NumMentions", 0) or 0)
        except (ValueError, TypeError):
            return False

        try:
            quad_class = int(row.get("QuadClass", 0) or 0)
        except (ValueError, TypeError):
            return False

        actor1_type = row.get("Actor1Type1Code", "") or ""
        actor2_type = row.get("Actor2Type1Code", "") or ""
        event_root = str(row.get("EventRootCode", "") or "").zfill(2)

        is_conflict = quad_class >= MIN_QUADCLASS_CONFLICT and num_mentions >= self.min_mentions
        is_media_event = ("MED" in (actor1_type, actor2_type)) and num_mentions >= 5
        is_infowar_code = event_root in INFOWAR_ROOT_CODES and num_mentions >= self.min_mentions

        return is_conflict or is_media_event or is_infowar_code

    async def _process_row(self, row: dict, ts_str: str, stats: IngestStats) -> None:
        event_id = row.get("GlobalEventID", "")
        doc, event_unix = build_document(row)
        if not doc.strip():
            stats.records_skipped += 1
            return

        stats.records_processed += 1

        try:
            quad_class = int(row.get("QuadClass", 0) or 0)
            num_mentions = int(row.get("NumMentions", 0) or 0)
            avg_tone = float(row.get("AvgTone", 0.0) or 0.0)
        except (ValueError, TypeError):
            quad_class, num_mentions, avg_tone = 0, 0, 0.0

        actor1 = row.get("Actor1Name", "") or ""
        actor2 = row.get("Actor2Name", "") or ""
        action_country = row.get("ActionGeo_CountryCode", "") or ""
        action_geo = row.get("ActionGeo_FullName", "") or ""
        event_code = row.get("EventCode", "") or ""
        day = row.get("Day", "") or ""

        pub_date = f"{day[:4]}-{day[4:6]}-{day[6:8]}" if len(day) == 8 else ""
        entity_tags = [t for t in [actor1, actor2, action_country, action_geo, event_code] if t]
        ingest_unix = event_unix or int(time.time())

        point_id = str(uuid.UUID(bytes=hashlib.sha256(f"gdelt:{event_id}:{ts_str}".encode()).digest()[:16]))
        payload: dict = {
            "text": doc,
            "source": SOURCE_LABEL,
            "document_type": "global_event",
            "provenance": "gdelt2",
            "ingest_date": TODAY,
            "event_id": event_id,
            "file_ts": ts_str,
            "quad_class": quad_class,
            "num_mentions": num_mentions,
            "avg_tone": round(avg_tone, 3),
            "pub_date": pub_date,
            "entity_tags": entity_tags,
            "ingested_at_unix": ingest_unix,
        }
        if action_country:
            payload["country"] = action_country

        self._upsert_buffer.append(
            qdrant_models.PointStruct(id=point_id, vector=[0.0] * EMBEDDING_DIM, payload=payload)
        )

        if len(self._upsert_buffer) >= self.upsert_batch_size:
            await self._flush_upsert_buffer(stats)

        if self.enqueue_notable and quad_class >= NOTABLE_MIN_QUADCLASS and num_mentions >= NOTABLE_MIN_MENTIONS:
            await self._maybe_enqueue(event_id, row, stats)

    async def _maybe_enqueue(self, event_id: str, row: dict, stats: IngestStats) -> None:
        if not self._redis or self.dry_run:
            return
        redis_key = f"osia:gdelt:enqueued:{event_id}"
        if await self._redis.exists(redis_key):
            return

        actor1 = row.get("Actor1Name", "") or ""
        actor2 = row.get("Actor2Name", "") or ""
        action_geo = row.get("ActionGeo_FullName", "") or ""
        num_mentions = row.get("NumMentions", "")
        source_url = row.get("SOURCEURL", "") or ""
        day = row.get("Day", "") or ""
        pub_date = f"{day[:4]}-{day[4:6]}-{day[6:8]}" if len(day) == 8 else day

        topic = f"GDELT high-impact conflict: {actor1} vs {actor2} in {action_geo} ({pub_date})"
        topic += f" — {num_mentions} media mentions"
        if source_url:
            topic += f" — source: {source_url[:120]}"

        job = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": topic,
                "desk": "information-warfare-desk",
                "priority": "normal",
                "triggered_by": "gdelt_ingest",
                "metadata": {
                    "event_id": event_id,
                    "actor1": actor1,
                    "actor2": actor2,
                    "location": action_geo,
                    "source_url": source_url,
                },
            }
        )
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        await self._redis.set(redis_key, "1", ex=60 * 60 * 24 * 14)
        stats.events_enqueued += 1

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
                        break
                except Exception as exc:
                    logger.warning("Embed attempt %d failed: %s", attempt + 1, exc)
                    await asyncio.sleep(5 * (attempt + 1))
        return [[0.0] * EMBEDDING_DIM for _ in texts]

    async def _save_checkpoint(self, ts_str: str) -> None:
        if self.dry_run or not self._redis:
            return
        await self._redis.set(CHECKPOINT_KEY, ts_str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest GDELT 2.0 global events into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="Skip Qdrant writes and Redis updates")
    p.add_argument("--resume", action="store_true", help="Resume from last Redis checkpoint")
    p.add_argument("--days-back", type=int, default=7, help="Days to backfill on first run")
    p.add_argument("--min-mentions", type=int, default=10, help="Minimum NumMentions for conflict events")
    p.add_argument("--enqueue-notable", action="store_true", help="Push high-coverage events to research queue")
    p.add_argument("--limit", type=int, default=0, help="Stop after N events ingested (0=no limit)")
    p.add_argument("--embed-batch-size", type=int, default=48, help="Texts per HF embedding call")
    p.add_argument("--embed-concurrency", type=int, default=3, help="Parallel embedding calls")
    p.add_argument("--upsert-batch-size", type=int, default=64, help="Points per Qdrant upsert call")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    ingestor = GdeltIngestor(args)
    asyncio.run(ingestor.run())
