"""
OSIA NOAA Storm Events Database Ingestion

Downloads and ingests the NOAA Storm Events Database — the official US record of
significant weather events, including tornadoes, hurricanes, floods, droughts,
heat waves, wildfires, and severe thunderstorms, with damage estimates, fatality
counts, and narrative descriptions of each event.

Source: https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
Free public data, no authentication required.

Each event record includes:
  - Event type, state, county, begin/end times
  - Death, injury, and damage figures (property and crop)
  - Free-text event narrative and episode narrative
  - Magnitude data where applicable (wind speed, hail size, etc.)

Usage:
  uv run python scripts/ingest_noaa_storm_events.py
  uv run python scripts/ingest_noaa_storm_events.py --dry-run
  uv run python scripts/ingest_noaa_storm_events.py --resume
  uv run python scripts/ingest_noaa_storm_events.py --year-from 2010 --year-to 2024
  uv run python scripts/ingest_noaa_storm_events.py --enqueue-notable
  uv run python scripts/ingest_noaa_storm_events.py --event-types "Hurricane" "Drought" "Wildfire"

Options:
  --dry-run             Parse and embed but skip Qdrant writes and Redis updates
  --resume              Skip years already completed in Redis checkpoint set
  --year-from Y         First year to ingest (default: 2010)
  --year-to Y           Last year to ingest (default: current year)
  --event-types         Only ingest these event types (default: all high-impact types)
  --all-event-types     Ingest all event types, not just high-impact ones
  --enqueue-notable     Push catastrophic events to Environment desk research queue
  --data-dir            Local cache dir for downloaded CSV files (default: /tmp/noaa_storm_events)
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
import gzip
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
import redis.asyncio as aioredis
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("osia.noaa_storm_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "noaa-storm-events"
EMBEDDING_DIM = 384
SOURCE_LABEL = "NOAA Storm Events Database"

NOAA_INDEX_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
USER_AGENT = "OSIA-Framework/1.0 (open-source intelligence research; +https://osia.dev)"
DOWNLOAD_DELAY = 1.5  # seconds between file downloads

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

COMPLETED_YEARS_KEY = "osia:noaa_storm:completed_years"
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")
CURRENT_YEAR = datetime.now(UTC).year

# High-impact event types that are intelligence-relevant
HIGH_IMPACT_TYPES = {
    "Hurricane",
    "Typhoon",
    "Tropical Storm",
    "Tropical Depression",
    "Drought",
    "Excessive Heat",
    "Heat",
    "Wildfire",
    "Flood",
    "Flash Flood",
    "Coastal Flood",
    "Lakeshore Flood",
    "Tornado",
    "Tsunami",
    "Storm Surge/Tide",
    "Blizzard",
    "Ice Storm",
    "Winter Storm",
    "High Wind",
    "Dust Storm",
    "Debris Flow",
    "Avalanche",
    "Landslide",
    "Volcanic Ash",
}

# Catastrophic threshold (property damage in millions USD) for enqueue-notable
CATASTROPHIC_DAMAGE_THRESHOLD = 1_000_000_000  # $1B

# Damage multipliers for NOAA's coded values (K=thousands, M=millions, B=billions)
_DAMAGE_MULT = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


# ---------------------------------------------------------------------------
# NOAA file index parser
# ---------------------------------------------------------------------------


def _parse_damage(val: str) -> float:
    """Convert NOAA damage field like '1.5M' or '250K' to float dollars."""
    val = (val or "").strip().upper()
    if not val or val in ("0", ""):
        return 0.0
    suffix = val[-1]
    if suffix in _DAMAGE_MULT:
        try:
            return float(val[:-1]) * _DAMAGE_MULT[suffix]
        except ValueError:
            return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0


def _month_int(yearmonth: str) -> int:
    """Extract month from YYYYMM string."""
    try:
        return int(str(yearmonth)[4:6])
    except (ValueError, IndexError):
        return 0


def parse_noaa_file_index(html: str) -> list[tuple[int, str]]:
    """
    Parse the NOAA directory listing HTML to extract detail CSV filenames.
    Returns list of (year, filename) sorted by year descending.
    """
    # Match filenames like: StormEvents_details-ftp_v1.0_d2024_c20250212.csv.gz
    pattern = re.compile(r'(StormEvents_details-ftp_v[\d.]+_d(\d{4})_c\d{8}\.csv\.gz)"')
    seen_years: dict[int, str] = {}
    for fname, year_str in pattern.findall(html):
        year = int(year_str)
        # If multiple versions exist for the same year, take the latest (last one in listing)
        seen_years[year] = fname
    return sorted(seen_years.items(), reverse=True)


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


def _safe(val: object, default: str = "") -> str:
    if val is None:
        return default
    s = str(val).strip()
    return s if s not in ("", "nan", "None", "NaN") else default


def build_document(row: dict) -> tuple[str, int | None]:
    """
    Build a narrative text document from a NOAA storm event row.
    Returns (document_text, event_unix_timestamp).
    """
    event_type = _safe(row.get("EVENT_TYPE"))
    state = _safe(row.get("STATE"))
    county = _safe(row.get("CZ_NAME"))
    begin_ym = _safe(row.get("BEGIN_YEARMONTH"))
    begin_day = _safe(row.get("BEGIN_DAY"))
    end_ym = _safe(row.get("END_YEARMONTH"))
    end_day = _safe(row.get("END_DAY"))
    event_narrative = _safe(row.get("EVENT_NARRATIVE"))
    episode_narrative = _safe(row.get("EPISODE_NARRATIVE"))
    deaths_direct = _safe(row.get("DEATHS_DIRECT", "0"))
    deaths_indirect = _safe(row.get("DEATHS_INDIRECT", "0"))
    injuries_direct = _safe(row.get("INJURIES_DIRECT", "0"))
    damage_property = _safe(row.get("DAMAGE_PROPERTY", "0"))
    damage_crops = _safe(row.get("DAMAGE_CROPS", "0"))
    magnitude = _safe(row.get("MAGNITUDE"))
    magnitude_type = _safe(row.get("MAGNITUDE_TYPE"))
    begin_lat = _safe(row.get("BEGIN_LAT"))
    begin_lon = _safe(row.get("BEGIN_LON"))
    source = _safe(row.get("SOURCE"))

    # Parse event date from YYYYMM + day
    event_unix: int | None = None
    if begin_ym and len(begin_ym) >= 6:
        try:
            year = int(begin_ym[:4])
            month = int(begin_ym[4:6])
            day = int(begin_day) if begin_day else 1
            dt = datetime(year, month, min(day, 28), tzinfo=UTC)
            event_unix = int(dt.timestamp())
        except (ValueError, OverflowError):
            event_unix = None  # malformed date fields — leave timestamp unset

    if not event_type and not event_narrative:
        return "", event_unix

    prop_dmg = _parse_damage(damage_property)
    crop_dmg = _parse_damage(damage_crops)
    total_dmg = prop_dmg + crop_dmg
    total_deaths = (int(float(deaths_direct)) if deaths_direct else 0) + (
        int(float(deaths_indirect)) if deaths_indirect else 0
    )
    total_injuries = int(float(injuries_direct)) if injuries_direct else 0

    # Build date string
    date_str = ""
    if begin_ym and len(begin_ym) >= 6:
        try:
            date_str = f"{begin_ym[:4]}-{begin_ym[4:6]}"
            if begin_day:
                date_str += f"-{int(begin_day):02d}"
        except ValueError:
            date_str = begin_ym  # fall back to raw YYYYMM string

    lines: list[str] = []
    lines.append(f"Event Type: {event_type}")
    lines.append(f"Location: {county}, {state}")
    if date_str:
        lines.append(f"Begin Date: {date_str}")
    if end_ym:
        try:
            end_str = f"{end_ym[:4]}-{end_ym[4:6]}"
            if end_day:
                end_str += f"-{int(end_day):02d}"
            if end_str != date_str:
                lines.append(f"End Date: {end_str}")
        except ValueError:
            pass  # end date fields malformed — omit end date from document

    if magnitude and magnitude_type:
        lines.append(f"Magnitude: {magnitude} {magnitude_type}")
    elif magnitude:
        lines.append(f"Magnitude: {magnitude}")

    if total_deaths > 0:
        lines.append(f"Fatalities: {total_deaths} (direct: {deaths_direct}, indirect: {deaths_indirect})")
    if total_injuries > 0:
        lines.append(f"Injuries: {total_injuries}")
    if total_dmg > 0:
        lines.append(f"Estimated Damage: ${total_dmg:,.0f} (property: ${prop_dmg:,.0f} / crops: ${crop_dmg:,.0f})")
    if begin_lat and begin_lon:
        lines.append(f"Coordinates: {begin_lat}, {begin_lon}")
    if source:
        lines.append(f"Source: {source}")

    if episode_narrative:
        lines.append(f"\nEpisode Context:\n{episode_narrative}")
    if event_narrative:
        lines.append(f"\nEvent Narrative:\n{event_narrative}")

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
    points_upserted: int = 0
    reports_enqueued: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> str:
        secs = int(time.time() - self.started_at)
        return f"{secs // 60}m{secs % 60:02d}s"

    def log_progress(self) -> None:
        logger.info(
            "seen=%d processed=%d skipped=%d upserted=%d enqueued=%d errors=%d elapsed=%s",
            self.records_seen,
            self.records_processed,
            self.records_skipped,
            self.points_upserted,
            self.reports_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class NoaaStormIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.enqueue_notable: bool = args.enqueue_notable
        self.year_from: int = args.year_from
        self.year_to: int = args.year_to
        self.resume: bool = args.resume
        self.data_dir: Path = Path(args.data_dir)
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size

        if args.all_event_types:
            self.event_type_filter: set[str] | None = None
        elif args.event_types:
            self.event_type_filter = {t.strip() for t in args.event_types}
        else:
            self.event_type_filter = HIGH_IMPACT_TYPES

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    async def run(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await self._ensure_collection()
            stats = IngestStats()

            # Get file index from NOAA
            async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=30.0) as http:
                resp = await http.get(NOAA_INDEX_URL)
                resp.raise_for_status()
                year_files = parse_noaa_file_index(resp.text)

            logger.info("NOAA index: found %d annual detail files", len(year_files))

            for year, filename in year_files:
                if year < self.year_from or year > self.year_to:
                    continue

                if self.resume and await self._is_year_complete(year):
                    logger.info("Skipping year %d (already complete).", year)
                    continue

                logger.info("Processing year %d (%s)...", year, filename)
                try:
                    await self._ingest_year(year, filename, stats)
                    await self._flush_upsert_buffer(stats)
                    await self._mark_year_complete(year)
                    stats.log_progress()
                except Exception as exc:
                    stats.errors += 1
                    logger.error("Failed to ingest year %d: %s", year, exc)

                await asyncio.sleep(DOWNLOAD_DELAY)

            await self._flush_upsert_buffer(stats)
            stats.log_progress()
            logger.info("NOAA Storm Events ingestion complete.")
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
        keyword_fields = ["event_type", "state", "pub_date", "document_type", "provenance"]
        float_fields = ["damage_usd", "damage_weight", "ingested_at_unix"]
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

    async def _ingest_year(self, year: int, filename: str, stats: IngestStats) -> None:
        """Download, decompress, and ingest one year's storm events CSV."""
        cache_path = self.data_dir / filename
        url = NOAA_INDEX_URL + filename

        if not cache_path.exists():
            logger.info("  Downloading %s ...", url)
            async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=120.0) as http:
                for attempt in range(3):
                    try:
                        resp = await http.get(url, follow_redirects=True)
                        resp.raise_for_status()
                        cache_path.write_bytes(resp.content)
                        logger.info("  Downloaded %.1f MB", len(resp.content) / (1024 * 1024))
                        break
                    except Exception as exc:
                        logger.warning("  Download attempt %d failed: %s", attempt + 1, exc)
                        await asyncio.sleep(10 * (attempt + 1))
                else:
                    raise RuntimeError(f"Failed to download {url}")
        else:
            logger.info("  Using cached file: %s", cache_path)

        # Decompress and parse in executor (CPU-bound)
        rows = await asyncio.get_event_loop().run_in_executor(None, self._parse_csv_gz, cache_path)
        logger.info("  Parsed %d rows from year %d", len(rows), year)

        for row in rows:
            stats.records_seen += 1
            event_type = _safe(row.get("EVENT_TYPE", ""))
            if self.event_type_filter and event_type not in self.event_type_filter:
                stats.records_skipped += 1
                continue

            try:
                await self._process_row(row, stats)
            except Exception as exc:
                stats.errors += 1
                logger.debug("Error processing row: %s", exc)

            if stats.records_processed % 10_000 == 0 and stats.records_processed > 0:
                stats.log_progress()

    def _parse_csv_gz(self, path: Path) -> list[dict]:
        """Decompress and parse a NOAA storm events gzipped CSV."""
        with gzip.open(path, "rt", encoding="latin-1", errors="replace") as fh:
            return list(csv.DictReader(fh))

    async def _process_row(self, row: dict, stats: IngestStats) -> None:
        event_id = _safe(row.get("EVENT_ID", ""))
        episode_id = _safe(row.get("EPISODE_ID", ""))
        state = _safe(row.get("STATE", ""))
        event_type = _safe(row.get("EVENT_TYPE", ""))

        doc, event_unix = build_document(row)
        if not doc.strip():
            stats.records_skipped += 1
            return

        stats.records_processed += 1

        prop_dmg = _parse_damage(_safe(row.get("DAMAGE_PROPERTY", "0")))
        crop_dmg = _parse_damage(_safe(row.get("DAMAGE_CROPS", "0")))
        total_dmg = prop_dmg + crop_dmg

        deaths = int(float(_safe(row.get("DEATHS_DIRECT", "0")) or "0")) + int(
            float(_safe(row.get("DEATHS_INDIRECT", "0")) or "0")
        )

        begin_ym = _safe(row.get("BEGIN_YEARMONTH", ""))
        pub_date = f"{begin_ym[:4]}-{begin_ym[4:6]}" if len(begin_ym) >= 6 else ""

        entity_tags = [t for t in [event_type, state, _safe(row.get("CZ_NAME", ""))] if t]
        ingest_unix = event_unix or int(time.time())

        # Normalised damage weight [0,1]: log1p-scaled, ceiling at $10B damage.
        # Used by the orchestrator as a score multiplier for boost-collection queries.
        damage_weight = min(1.0, math.log1p(total_dmg) / math.log1p(10_000_000_000)) if total_dmg > 0 else 0.0

        point_id = str(uuid.UUID(bytes=hashlib.sha256(f"noaa:storm:{event_id}".encode()).digest()[:16]))
        payload: dict = {
            "text": doc,
            "source": SOURCE_LABEL,
            "document_type": "storm_event",
            "provenance": "noaa_storm_events",
            "ingest_date": TODAY,
            "event_id": event_id,
            "episode_id": episode_id,
            "event_type": event_type,
            "state": state,
            "pub_date": pub_date,
            "entity_tags": entity_tags,
            "ingested_at_unix": ingest_unix,
            "damage_weight": damage_weight,
        }
        if total_dmg > 0:
            payload["damage_usd"] = total_dmg
        if deaths > 0:
            payload["deaths"] = deaths

        self._upsert_buffer.append(
            qdrant_models.PointStruct(id=point_id, vector=[0.0] * EMBEDDING_DIM, payload=payload)
        )

        if len(self._upsert_buffer) >= self.upsert_batch_size:
            await self._flush_upsert_buffer(stats)

        if self.enqueue_notable and (total_dmg >= CATASTROPHIC_DAMAGE_THRESHOLD or deaths >= 50):
            await self._maybe_enqueue(event_id, event_type, state, total_dmg, deaths, pub_date, stats)

    async def _maybe_enqueue(
        self,
        event_id: str,
        event_type: str,
        state: str,
        damage: float,
        deaths: int,
        pub_date: str,
        stats: IngestStats,
    ) -> None:
        if not self._redis or self.dry_run:
            return
        redis_key = f"osia:noaa:enqueued:{event_id}"
        if await self._redis.exists(redis_key):
            return
        topic = f"NOAA {event_type} in {state}"
        if pub_date:
            topic += f" ({pub_date})"
        if damage >= CATASTROPHIC_DAMAGE_THRESHOLD:
            topic += f" — ${damage / 1e9:.1f}B damage"
        if deaths >= 50:
            topic += f" — {deaths} fatalities"
        job = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": topic,
                "desk": "environment-and-ecology-desk",
                "priority": "normal",
                "triggered_by": "noaa_storm_ingest",
                "metadata": {"event_id": event_id, "event_type": event_type, "state": state, "damage_usd": damage},
            }
        )
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        await self._redis.set(redis_key, "1", ex=60 * 60 * 24 * 60)
        stats.reports_enqueued += 1
        logger.debug("Enqueued research: %r", topic)

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

    async def _mark_year_complete(self, year: int) -> None:
        if self.dry_run or not self._redis:
            return
        await self._redis.sadd(COMPLETED_YEARS_KEY, str(year))

    async def _is_year_complete(self, year: int) -> bool:
        if not self._redis:
            return False
        return bool(await self._redis.sismember(COMPLETED_YEARS_KEY, str(year)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest NOAA Storm Events Database into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="Skip Qdrant writes and Redis updates")
    p.add_argument("--resume", action="store_true", help="Skip years already completed")
    p.add_argument("--year-from", type=int, default=2010, help="First year to ingest")
    p.add_argument("--year-to", type=int, default=CURRENT_YEAR, help="Last year to ingest")
    p.add_argument("--event-types", nargs="+", help="Only ingest these event type names")
    p.add_argument("--all-event-types", action="store_true", help="Ingest all event types")
    p.add_argument("--enqueue-notable", action="store_true", help="Push catastrophic events to research queue")
    p.add_argument("--data-dir", default="/tmp/noaa_storm_events", help="Local cache directory")  # noqa: S108
    p.add_argument("--embed-batch-size", type=int, default=48, help="Texts per HF embedding call")
    p.add_argument("--embed-concurrency", type=int, default=3, help="Parallel embedding calls")
    p.add_argument("--upsert-batch-size", type=int, default=64, help="Points per Qdrant upsert call")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    ingestor = NoaaStormIngestor(args)
    asyncio.run(ingestor.run())
