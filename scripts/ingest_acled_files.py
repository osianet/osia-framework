"""
OSIA ACLED Local File Ingestion

Ingests ACLED conflict data from locally-downloaded Excel files in sources/acled/
into the 'acled-conflict-events' Qdrant collection.

Handles two file shapes exported from the ACLED data portal:

  Regional aggregated files (event-level weekly aggregates):
    Columns: WEEK, REGION, COUNTRY, ADMIN1, EVENT_TYPE, SUB_EVENT_TYPE,
             EVENTS, FATALITIES, POPULATION_EXPOSURE, DISORDER_TYPE,
             ID, CENTROID_LATITUDE, CENTROID_LONGITUDE

  Summary statistics files (country-year or country-month-year totals):
    Columns: COUNTRY, YEAR[, MONTH], EVENTS or FATALITIES

Usage:
  uv run python scripts/ingest_acled_files.py
  uv run python scripts/ingest_acled_files.py --dry-run
  uv run python scripts/ingest_acled_files.py --resume
  uv run python scripts/ingest_acled_files.py --limit 5000
  uv run python scripts/ingest_acled_files.py --enqueue-notable

Options:
  --dry-run             Parse and embed but skip Qdrant writes and Redis updates
  --resume              Skip files already marked complete in Redis
  --enqueue-notable     Push high-fatality weekly events to Geopolitical research queue
  --limit N             Stop after N rows total (0 = no limit)
  --embed-batch-size    Texts per HF embedding call (default: 32)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)
  --source-dir          Directory containing .xlsx files (default: sources/acled)

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
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import redis.asyncio as aioredis
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("osia.acled_files_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "acled-conflict-events"
EMBEDDING_DIM = 384
SOURCE_LABEL = "ACLED Armed Conflict Location & Event Data"
DEFAULT_SOURCE_DIR = "sources/acled"

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

# Redis keys
RESEARCH_QUEUE_KEY = "osia:research_queue"
FILE_DONE_KEY_PREFIX = "osia:acled_files:done:"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

NOTABLE_EVENT_TYPES = {"Battles", "Explosions/Remote violence", "Violence against civilians"}
NOTABLE_FATALITY_THRESHOLD = 10

# Regional file columns (all must be present)
REGIONAL_COLS = {"WEEK", "REGION", "COUNTRY", "ADMIN1", "EVENT_TYPE", "EVENTS", "FATALITIES"}

# Summary stat file columns
SUMMARY_COLS = {"COUNTRY", "YEAR"}


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------


def detect_file_type(df: pd.DataFrame) -> str:
    """Return 'regional' or 'summary' based on column presence."""
    cols = set(df.columns.str.upper())
    if REGIONAL_COLS.issubset(cols):
        return "regional"
    if SUMMARY_COLS.issubset(cols):
        return "summary"
    return "unknown"


def file_redis_key(path: Path) -> str:
    h = hashlib.md5(path.name.encode(), usedforsecurity=False).hexdigest()[:12]
    return f"{FILE_DONE_KEY_PREFIX}{h}"


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------


def _ts_from_week(week_val) -> int | None:
    """Convert pandas Timestamp or datetime to unix int."""
    try:
        if pd.isna(week_val):
            return None
        if hasattr(week_val, "timestamp"):
            return int(week_val.timestamp())
        dt = pd.Timestamp(week_val)
        return int(dt.timestamp())
    except Exception:
        return None


def _ts_from_year_month(year, month=None) -> int | None:
    try:
        y = int(year)
        m = 1
        if month and not pd.isna(month):
            month_map = {
                "January": 1,
                "February": 2,
                "March": 3,
                "April": 4,
                "May": 5,
                "June": 6,
                "July": 7,
                "August": 8,
                "September": 9,
                "October": 10,
                "November": 11,
                "December": 12,
            }
            if isinstance(month, str):
                m = month_map.get(month, 1)
            else:
                m = int(month)
        return int(datetime(y, m, 1, tzinfo=UTC).timestamp())
    except Exception:
        return None


def build_regional_document(row: dict, filename: str) -> tuple[str, int | None]:
    """Build narrative text from a regional aggregated event row."""
    week = row.get("WEEK")
    region = row.get("REGION", "")
    country = row.get("COUNTRY", "")
    admin1 = row.get("ADMIN1", "")
    event_type = row.get("EVENT_TYPE", "")
    sub_event_type = row.get("SUB_EVENT_TYPE", "")
    disorder_type = row.get("DISORDER_TYPE", "")
    events = row.get("EVENTS")
    fatalities = row.get("FATALITIES")
    pop_exposure = row.get("POPULATION_EXPOSURE")
    lat = row.get("CENTROID_LATITUDE")
    lon = row.get("CENTROID_LONGITUDE")

    week_str = ""
    event_unix = None
    if week is not None and not (isinstance(week, float) and pd.isna(week)):
        try:
            ts = pd.Timestamp(week)
            week_str = ts.strftime("%Y-%m-%d")
            event_unix = int(ts.timestamp())
        except Exception:
            pass

    # Pre-compute counts so the narrative sentence can use them
    try:
        ev = int(float(events)) if events is not None else 0
    except (TypeError, ValueError):
        ev = 0
    try:
        fat = int(float(fatalities)) if fatalities is not None else 0
    except (TypeError, ValueError):
        fat = 0

    location_parts = [p for p in [admin1, country, region] if p and not (isinstance(p, float) and pd.isna(p))]

    lines: list[str] = []

    # Lead with a prose narrative sentence so the embedding model has natural language
    # to anchor semantic queries on — not just structured key:value fields.
    if event_type and week_str and location_parts:
        ev_str = f"{ev} " if ev > 0 else ""
        fat_suffix = f", resulting in {fat} {'fatality' if fat == 1 else 'fatalities'}" if fat > 0 else ""
        lines.append(
            f"In the week of {week_str}, ACLED recorded {ev_str}"
            f"{event_type.lower()} events in {', '.join(location_parts)}{fat_suffix}."
        )

    if event_type:
        lines.append(f"Event Type: {event_type}" + (f" — {sub_event_type}" if sub_event_type else ""))
    if disorder_type:
        lines.append(f"Disorder Type: {disorder_type}")
    if week_str:
        lines.append(f"Week of: {week_str}")

    if location_parts:
        lines.append(f"Location: {', '.join(location_parts)}")

    try:
        if lat is not None and lon is not None and not pd.isna(lat) and not pd.isna(lon):
            lines.append(f"Coordinates: {float(lat):.4f}, {float(lon):.4f}")
    except (TypeError, ValueError):
        pass

    if ev > 0:
        lines.append(f"Events this week: {ev}")
    if fat > 0:
        lines.append(f"Fatalities: {fat}")

    try:
        pop = int(float(pop_exposure)) if pop_exposure is not None else 0
        if pop > 0:
            lines.append(f"Population Exposure: {pop:,}")
    except (TypeError, ValueError):
        pass

    if len(lines) < 3:
        return "", event_unix

    return "\n".join(lines), event_unix


def build_summary_document(row: dict, stat_label: str) -> tuple[str, int | None]:
    """Build a factual sentence from a summary statistics row."""
    country = row.get("COUNTRY", "")
    year = row.get("YEAR")
    month = row.get("MONTH")
    events = row.get("EVENTS")
    fatalities = row.get("FATALITIES")

    if not country or not year:
        return "", None

    event_unix = _ts_from_year_month(year, month)

    period = str(int(year))
    if month and not (isinstance(month, float) and pd.isna(month)):
        period = f"{month} {int(year)}"

    # Pre-compute values for narrative
    ev_int: int | None = None
    fat_int: int | None = None
    if events is not None and not (isinstance(events, float) and pd.isna(events)):
        try:
            ev_int = int(float(events))
        except (TypeError, ValueError):
            pass
    if fatalities is not None and not (isinstance(fatalities, float) and pd.isna(fatalities)):
        try:
            fat_int = int(float(fatalities))
        except (TypeError, ValueError):
            pass

    lines: list[str] = []

    # Lead with a prose narrative sentence for better semantic embedding
    ev_str = f"{ev_int} events" if ev_int is not None else "events"
    fat_str = f" and {fat_int} fatalities" if fat_int is not None else ""
    lines.append(f"{country} reported {ev_str}{fat_str} in {period} according to ACLED ({stat_label}).")

    lines.append(f"ACLED Statistics — {stat_label}")
    lines.append(f"Country: {country}")
    lines.append(f"Period: {period}")
    if ev_int is not None:
        lines.append(f"Events: {ev_int}")
    if fat_int is not None:
        lines.append(f"Fatalities: {fat_int}")

    if len(lines) < 3:
        return "", event_unix

    return "\n".join(lines), event_unix


def _stat_label_from_filename(name: str) -> str:
    """Derive a human-readable label from the summary file name."""
    stem = Path(name).stem
    # Strip trailing date suffix like "_as-of-03Apr2026"
    stem = re.sub(r"_as-of-\w+$", "", stem)
    stem = stem.replace("_", " ").replace("-", " ")
    # Title-case key words
    return stem.title()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    files_processed: int = 0
    files_skipped: int = 0
    rows_seen: int = 0
    rows_skipped: int = 0
    rows_processed: int = 0
    points_upserted: int = 0
    events_enqueued: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> str:
        secs = int(time.time() - self.started_at)
        return f"{secs // 60}m{secs % 60:02d}s"

    def log_progress(self) -> None:
        logger.info(
            "files=%d/%d rows_seen=%d processed=%d skipped=%d upserted=%d enqueued=%d errors=%d elapsed=%s",
            self.files_processed,
            self.files_processed + self.files_skipped,
            self.rows_seen,
            self.rows_processed,
            self.rows_skipped,
            self.points_upserted,
            self.events_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class AcledFileIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.resume: bool = args.resume
        self.enqueue_notable: bool = args.enqueue_notable
        self.limit: int = args.limit
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.source_dir: Path = Path(args.source_dir)

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    async def run(self) -> None:
        if not HF_TOKEN:
            logger.error("HF_TOKEN must be set in .env")
            return

        xlsx_files = sorted(self.source_dir.glob("*.xlsx"))
        if not xlsx_files:
            logger.error("No .xlsx files found in %s", self.source_dir)
            return

        logger.info("Found %d Excel files in %s", len(xlsx_files), self.source_dir)

        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        stats = IngestStats()
        try:
            await self._ensure_collection()

            for path in xlsx_files:
                if self.limit and stats.rows_processed >= self.limit:
                    logger.info("Reached --limit %d — stopping.", self.limit)
                    break

                if self.resume and self._redis:
                    done = await self._redis.exists(file_redis_key(path))
                    if done:
                        logger.info("Skipping (already done): %s", path.name)
                        stats.files_skipped += 1
                        continue

                await self._ingest_file(path, stats)

            await self._flush_upsert_buffer(stats)
            stats.log_progress()
            logger.info("ACLED file ingestion complete.")
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

        # Ensure payload indexes exist (idempotent — safe to call on every run).
        keyword_fields = ["event_type", "country", "region", "document_type", "provenance"]
        float_fields = ["fatality_weight", "ingested_at_unix"]
        for field_name in keyword_fields:
            await self._qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
        for field_name in float_fields:
            await self._qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=qdrant_models.PayloadSchemaType.FLOAT,
            )
        logger.info("Payload indexes verified for '%s'.", COLLECTION_NAME)

    async def _ingest_file(self, path: Path, stats: IngestStats) -> None:
        logger.info("Reading %s ...", path.name)
        try:
            df = pd.read_excel(path)
        except Exception as exc:
            logger.error("Failed to read %s: %s", path.name, exc)
            stats.errors += 1
            return

        # Normalise column names to upper-case for detection
        df.columns = df.columns.str.upper()
        file_type = detect_file_type(df)
        logger.info("%s → %s (%d rows)", path.name, file_type, len(df))

        if file_type == "unknown":
            logger.warning("Unrecognised column layout in %s — skipping.", path.name)
            stats.files_skipped += 1
            return

        stat_label = _stat_label_from_filename(path.name) if file_type == "summary" else ""
        rows_before = stats.rows_processed

        for idx, row in df.iterrows():
            stats.rows_seen += 1

            try:
                if file_type == "regional":
                    doc, event_unix = build_regional_document(row.to_dict(), path.name)
                else:
                    doc, event_unix = build_summary_document(row.to_dict(), stat_label)
            except Exception as exc:
                stats.errors += 1
                logger.debug("Row %d in %s: build error: %s", idx, path.name, exc)
                continue

            if not doc.strip():
                stats.rows_skipped += 1
                continue

            stats.rows_processed += 1
            point_id = str(uuid.UUID(bytes=hashlib.sha256(f"acled_file:{path.name}:{idx}".encode()).digest()[:16]))
            ingest_unix = event_unix or int(time.time())

            payload: dict = {
                "text": doc,
                "source": SOURCE_LABEL,
                "document_type": "conflict_event" if file_type == "regional" else "conflict_statistics",
                "provenance": "acled_files",
                "source_file": path.name,
                "ingest_date": TODAY,
                "ingested_at_unix": ingest_unix,
                "entity_tags": self._extract_entity_tags(row.to_dict(), file_type),
            }

            if file_type == "regional":
                for field_name, col in [("event_type", "EVENT_TYPE"), ("country", "COUNTRY"), ("region", "REGION")]:
                    val = row.get(col)
                    if val and not (isinstance(val, float) and pd.isna(val)):
                        payload[field_name] = str(val)
                try:
                    fat = int(float(row.get("FATALITIES", 0) or 0))
                    if fat > 0:
                        payload["fatalities"] = fat
                except (TypeError, ValueError):
                    fat = 0
                # Normalised fatality weight [0,1]: log1p-scaled, ceiling at 100 fatalities.
                # Used by the orchestrator as a score multiplier for boost-collection queries.
                payload["fatality_weight"] = min(1.0, math.log1p(fat) / math.log1p(100))

            self._upsert_buffer.append(
                qdrant_models.PointStruct(id=point_id, vector=[0.0] * EMBEDDING_DIM, payload=payload)
            )

            if len(self._upsert_buffer) >= self.upsert_batch_size:
                await self._flush_upsert_buffer(stats)

            if self.enqueue_notable and file_type == "regional" and self._is_notable(row.to_dict()):
                await self._maybe_enqueue(row.to_dict(), stats)

            if self.limit and stats.rows_processed >= self.limit:
                break

            if stats.rows_processed % 5000 == 0 and stats.rows_processed > rows_before:
                stats.log_progress()

        # Flush any remaining rows from this file before marking done
        await self._flush_upsert_buffer(stats)

        stats.files_processed += 1
        logger.info(
            "Finished %s — %d rows processed this file.",
            path.name,
            stats.rows_processed - rows_before,
        )

        if not self.dry_run and self._redis:
            await self._redis.set(file_redis_key(path), "1")

    def _extract_entity_tags(self, row: dict, file_type: str) -> list[str]:
        tags: list[str] = []
        if file_type == "regional":
            for col in ("EVENT_TYPE", "COUNTRY", "ADMIN1", "REGION", "DISORDER_TYPE", "SUB_EVENT_TYPE"):
                val = row.get(col)
                if val and not (isinstance(val, float) and pd.isna(val)):
                    tags.append(str(val))
        else:
            for col in ("COUNTRY",):
                val = row.get(col)
                if val and not (isinstance(val, float) and pd.isna(val)):
                    tags.append(str(val))
        return [t for t in tags if t]

    def _is_notable(self, row: dict) -> bool:
        try:
            fat = int(float(row.get("FATALITIES", 0) or 0))
        except (TypeError, ValueError):
            fat = 0
        event_type = str(row.get("EVENT_TYPE", ""))
        return fat >= NOTABLE_FATALITY_THRESHOLD or event_type in NOTABLE_EVENT_TYPES

    async def _maybe_enqueue(self, row: dict, stats: IngestStats) -> None:
        if not self._redis or self.dry_run:
            return
        try:
            week = row.get("WEEK")
            week_str = pd.Timestamp(week).strftime("%Y-%m-%d") if week is not None else "unknown"
            country = str(row.get("COUNTRY", ""))
            event_type = str(row.get("EVENT_TYPE", ""))
            fat = int(float(row.get("FATALITIES", 0) or 0))

            dedup_key = f"osia:acled_files:enqueued:{country}:{event_type}:{week_str}"
            if await self._redis.exists(dedup_key):
                return

            topic = f"ACLED: {event_type} in {country} (week of {week_str})"
            if fat >= NOTABLE_FATALITY_THRESHOLD:
                topic += f" — {fat} fatalities"

            job = json.dumps(
                {
                    "job_id": str(uuid.uuid4()),
                    "topic": topic,
                    "desk": "geopolitical-and-security-desk",
                    "priority": "normal",
                    "triggered_by": "acled_files_ingest",
                    "metadata": {"country": country, "event_type": event_type, "fatalities": fat},
                }
            )
            await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
            await self._redis.set(dedup_key, "1", ex=60 * 60 * 24 * 30)
            stats.events_enqueued += 1
        except Exception as exc:
            logger.debug("Enqueue error: %s", exc)

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest local ACLED Excel files into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="Skip Qdrant writes and Redis updates")
    p.add_argument("--resume", action="store_true", help="Skip files already marked complete in Redis")
    p.add_argument("--enqueue-notable", action="store_true", help="Push high-fatality events to research queue")
    p.add_argument("--limit", type=int, default=0, help="Stop after N rows total (0=no limit)")
    p.add_argument("--embed-batch-size", type=int, default=32, help="Texts per HF embedding call")
    p.add_argument("--embed-concurrency", type=int, default=3, help="Parallel embedding calls")
    p.add_argument("--upsert-batch-size", type=int, default=64, help="Points per Qdrant upsert call")
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR, help="Directory containing .xlsx files")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    ingestor = AcledFileIngestor(args)
    asyncio.run(ingestor.run())
