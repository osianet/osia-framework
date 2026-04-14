"""
OSIA EPA Toxic Release Inventory (TRI) Ingestion

Downloads and ingests the US EPA Toxics Release Inventory — the official database
of toxic chemical releases, disposals, and other waste management activities
reported by industrial facilities across the United States.

TRI is the primary public record for identifying which corporations are releasing
which toxic chemicals, in what quantities, and where — core intelligence for the
desk's "Corporate Ecocide Tracking" mandate.

Source: https://data.epa.gov/efservice/downloads/tri/mv_tri_basic_download/
Annual CSV downloads via EPA Envirofacts, free and public.

Each record covers:
  - Facility name, parent company, DUNS, geographic coordinates
  - Chemical name, CAS number, classification (carcinogen, PBT, etc.)
  - Total releases by media: air, water, land, underground injection
  - On-site disposal vs. off-site transfers
  - Year of report

Usage:
  uv run python scripts/ingest_epa_toxic_releases.py
  uv run python scripts/ingest_epa_toxic_releases.py --dry-run
  uv run python scripts/ingest_epa_toxic_releases.py --resume
  uv run python scripts/ingest_epa_toxic_releases.py --year-from 2015 --year-to 2023
  uv run python scripts/ingest_epa_toxic_releases.py --enqueue-notable
  uv run python scripts/ingest_epa_toxic_releases.py --min-release-lbs 10000

Options:
  --dry-run             Parse and embed but skip Qdrant writes and Redis updates
  --resume              Skip years already completed in Redis checkpoint set
  --year-from Y         First year to ingest (default: 2015)
  --year-to Y           Last year to ingest (default: 2023)
  --min-release-lbs N   Only ingest facilities releasing >= N lbs total (default: 1000)
  --enqueue-notable     Push top polluters to Environment desk research queue
  --data-dir            Local cache dir for downloaded CSV files (default: /tmp/epa_tri)
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
import math
import os
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
logger = logging.getLogger("osia.epa_tri_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "epa-toxic-releases"
EMBEDDING_DIM = 384
SOURCE_LABEL = "EPA Toxics Release Inventory (TRI)"

# EPA Envirofacts bulk CSV download — national data by year.
# Available years: 1987–present. Geography "US" = all US facilities.
TRI_CSV_URL_TEMPLATE = "https://data.epa.gov/efservice/downloads/tri/mv_tri_basic_download/{year}_US/csv"

USER_AGENT = "OSIA-Framework/1.0 (open-source intelligence research; +https://osia.dev)"
DOWNLOAD_DELAY = 2.0  # seconds between year downloads

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

COMPLETED_YEARS_KEY = "osia:epa_tri:completed_years"
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# Threshold for enqueue-notable (lbs of total releases in a single year)
NOTABLE_RELEASE_THRESHOLD_LBS = 1_000_000  # 1 million lbs


# ---------------------------------------------------------------------------
# TRI CSV column mapping (Envirofacts mv_tri_basic_download schema)
# ---------------------------------------------------------------------------

_COL_YEAR = "YEAR"
_COL_FACILITY = "FACILITY NAME"
_COL_PARENT = "PARENT CO NAME"
_COL_CITY = "CITY"
_COL_STATE = "ST"
_COL_COUNTY = "COUNTY"
_COL_LATITUDE = "LATITUDE"
_COL_LONGITUDE = "LONGITUDE"
_COL_INDUSTRY_SECTOR = "INDUSTRY SECTOR"
_COL_CHEMICAL = "CHEMICAL"
_COL_CAS = "CAS#"  # no space in Envirofacts schema (vs "CAS #" in old ZIP files)
_COL_CLASSIFICATION = "CLASSIFICATION"
_COL_UNIT = "UNIT OF MEASURE"
_COL_CARCINOGEN = "CARCINOGEN"

# Release columns — Envirofacts uses section numbers, no pre-computed media totals
_COL_FUGITIVE_AIR = "5.1 - FUGITIVE AIR"
_COL_STACK_AIR = "5.2 - STACK AIR"
_COL_WATER_RELEASES = "5.3 - WATER"
_COL_UNDERGROUND = "5.4 - UNDERGROUND"
_COL_ONSITE_TOTAL = "ON-SITE RELEASE TOTAL"
_COL_OFFSITE_TOTAL = "OFF-SITE RELEASE TOTAL"
_COL_TOTAL_RELEASES = "TOTAL RELEASES"  # grand total (on-site + off-site releases)


def _col(row: dict, *keys: str, default: str = "") -> str:
    """Try each key in order, return first non-empty match."""
    for key in keys:
        val = (row.get(key) or row.get(key.upper()) or row.get(key.lower()) or "").strip()
        if val and val not in (".", "NA", "N/A", "0"):
            return val
    return default


def _float(val: str, default: float = 0.0) -> float:
    try:
        return float(val.replace(",", "").strip())
    except (ValueError, AttributeError):
        return default


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


def build_document(row: dict) -> tuple[str, int | None]:
    """
    Build a structured text document for one TRI facility+chemical record.
    Returns (document_text, report_year_unix_timestamp).
    """
    year_str = _col(row, _COL_YEAR, "REPORTING YEAR")
    facility = _col(row, _COL_FACILITY, "FACILITY")
    parent = _col(row, _COL_PARENT, "PARENT COMPANY")
    city = _col(row, _COL_CITY)
    state = _col(row, _COL_STATE, "STATE")
    county = _col(row, _COL_COUNTY)
    lat = _col(row, _COL_LATITUDE, "LAT")
    lon = _col(row, _COL_LONGITUDE, "LONG")
    industry = _col(row, _COL_INDUSTRY_SECTOR, "INDUSTRY SECTOR CODE")
    chemical = _col(row, _COL_CHEMICAL, "CHEMICAL NAME")
    cas = _col(row, _COL_CAS, "CAS #", "CAS NUMBER")
    classification = _col(row, _COL_CLASSIFICATION)
    carcinogen = _col(row, _COL_CARCINOGEN)
    unit = _col(row, _COL_UNIT, "UNIT")

    # Air releases: sum fugitive + stack (Envirofacts splits what old ZIP had as one total)
    total_air = _float(_col(row, _COL_FUGITIVE_AIR)) + _float(_col(row, _COL_STACK_AIR))
    # Fallback: if old-format column present (cached files from before migration)
    if total_air == 0.0:
        total_air = _float(_col(row, "TOTAL AIR RELEASES", "AIR RELEASES"))

    total_water = _float(_col(row, _COL_WATER_RELEASES, "TOTAL WATER RELEASES", "WATER RELEASES"))
    total_underground = _float(
        _col(row, _COL_UNDERGROUND, "TOTAL UNDERGROUND INJECTION ON-SITE", "UNDERGROUND INJECTION")
    )
    total_onsite = _float(_col(row, _COL_ONSITE_TOTAL, "TOTAL RELEASES"))
    total_offsite = _float(_col(row, _COL_OFFSITE_TOTAL, "TOTAL TRANSFERS OFF-SITE", "OFF-SITE TRANSFERS"))
    total_all = _float(_col(row, _COL_TOTAL_RELEASES)) or (total_onsite + total_offsite)

    if not facility or not chemical:
        return "", None

    # Report year unix timestamp (Jan 1 of the report year)
    report_unix: int | None = None
    if year_str:
        try:
            report_unix = int(datetime(int(year_str), 1, 1, tzinfo=UTC).timestamp())
        except ValueError:
            report_unix = None

    lines: list[str] = []

    # Lead with a prose narrative sentence so the embedding model has natural language
    # to anchor semantic queries on — not just structured key:value fields.
    if facility and chemical and year_str:
        # Avoid "CORP (CORP)" when parent and facility names are identical
        corp = facility if not parent or parent.upper() == facility.upper() else f"{parent} ({facility})"
        location_str = f"{city}, {state}" if city else state
        narrative = f"In {year_str}, {corp} released {total_all:,.0f} lbs of {chemical} in {location_str}."
        if classification:
            narrative += f" Classified as {classification}."
        if carcinogen and carcinogen.upper() in ("YES", "Y"):
            narrative += " Listed carcinogen."
        lines.append(narrative)

    lines.append(f"Facility: {facility}")
    if parent:
        lines.append(f"Parent Corporation: {parent}")
    lines.append(f"Location: {city}, {county} County, {state}")
    if lat and lon:
        lines.append(f"Coordinates: {lat}, {lon}")
    if industry:
        lines.append(f"Industry Sector: {industry}")
    if year_str:
        lines.append(f"Reporting Year: {year_str}")

    lines.append("")
    lines.append(f"Chemical Released: {chemical}")
    if cas:
        lines.append(f"CAS Number: {cas}")
    if classification:
        lines.append(f"Classification: {classification}")
    if carcinogen and carcinogen.upper() in ("YES", "Y"):
        lines.append("Listed Carcinogen: Yes")
    if unit:
        lines.append(f"Unit of Measure: {unit}")

    lines.append("")
    lines.append("Release Quantities:")
    if total_air > 0:
        lines.append(f"  Air releases: {total_air:,.1f} {unit}")
    if total_water > 0:
        lines.append(f"  Water releases: {total_water:,.1f} {unit}")
    if total_underground > 0:
        lines.append(f"  Underground injection: {total_underground:,.1f} {unit}")
    if total_onsite > 0:
        lines.append(f"  Total on-site releases: {total_onsite:,.1f} {unit}")
    if total_offsite > 0:
        lines.append(f"  Off-site releases: {total_offsite:,.1f} {unit}")
    if total_all > 0:
        lines.append(f"  Combined total (on-site + off-site): {total_all:,.1f} {unit}")

    if len(lines) < 5:
        return "", report_unix

    return "\n".join(lines), report_unix


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    records_seen: int = 0
    records_skipped: int = 0
    records_processed: int = 0
    points_upserted: int = 0
    facilities_enqueued: int = 0
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
            self.facilities_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class EpaTRIIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.enqueue_notable: bool = args.enqueue_notable
        self.year_from: int = args.year_from
        self.year_to: int = args.year_to
        self.min_release_lbs: float = args.min_release_lbs
        self.resume: bool = args.resume
        self.data_dir: Path = Path(args.data_dir)
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size

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

            for data_year in range(self.year_from, self.year_to + 1):
                if self.resume and await self._is_year_complete(data_year):
                    logger.info("Skipping year %d (already complete).", data_year)
                    continue

                logger.info("Processing TRI year %d...", data_year)
                seen_before = stats.records_seen
                try:
                    await self._ingest_year(data_year, stats)
                    await self._flush_upsert_buffer(stats)
                    # Only mark complete if we actually read records from this year's file.
                    # Skipping when records_seen is unchanged prevents a zero-record run
                    # (e.g. column-name mismatch, empty download) from poisoning the checkpoint.
                    if stats.records_seen > seen_before:
                        await self._mark_year_complete(data_year)
                    else:
                        logger.warning("Year %d: no records seen — not marking complete.", data_year)
                    stats.log_progress()
                except Exception as exc:
                    stats.errors += 1
                    logger.error("Failed to ingest TRI year %d: %s", data_year, exc)

                await asyncio.sleep(DOWNLOAD_DELAY)

            await self._flush_upsert_buffer(stats)
            stats.log_progress()
            logger.info("EPA TRI ingestion complete.")
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
        keyword_fields = ["facility", "chemical", "state", "reporting_year", "document_type", "provenance"]
        float_fields = ["total_releases_lbs", "release_weight", "ingested_at_unix"]
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

    async def _ingest_year(self, data_year: int, stats: IngestStats) -> None:
        """Download the TRI CSV for one data year and ingest all records."""
        cache_path = self.data_dir / f"tri_{data_year}.csv"

        if not cache_path.exists():
            url = TRI_CSV_URL_TEMPLATE.format(year=data_year)
            downloaded = False
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT}, timeout=300.0, follow_redirects=True
            ) as http:
                for attempt in range(3):
                    try:
                        logger.info("  Downloading: %s", url)
                        # Stream to disk — national TRI files can exceed 50 MB
                        async with http.stream("GET", url) as resp:
                            if resp.status_code in (404, 410):
                                logger.warning(
                                    "  TRI data for year %d not available (%d) — skipping.",
                                    data_year,
                                    resp.status_code,
                                )
                                return
                            resp.raise_for_status()
                            with open(cache_path, "wb") as fh:
                                async for chunk in resp.aiter_bytes(65536):
                                    fh.write(chunk)
                        size_mb = cache_path.stat().st_size / (1024 * 1024)
                        logger.info("  Downloaded %.1f MB for year %d", size_mb, data_year)
                        downloaded = True
                        break
                    except Exception as exc:
                        logger.warning("  Attempt %d failed: %s", attempt + 1, exc)
                        if cache_path.exists():
                            cache_path.unlink()
                        await asyncio.sleep(10 * (attempt + 1))

            if not downloaded:
                logger.warning("Could not download TRI data for year %d — skipping.", data_year)
                return
        else:
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info("  Using cached file: %s (%.1f MB)", cache_path, size_mb)

        rows = await asyncio.get_event_loop().run_in_executor(None, self._parse_csv, cache_path, data_year)
        logger.info("  Parsed %d records from TRI year %d", len(rows), data_year)

        for row in rows:
            stats.records_seen += 1
            try:
                await self._process_row(row, stats)
            except Exception as exc:
                stats.errors += 1
                logger.debug("Error processing TRI row: %s", exc)

            if stats.records_processed % 10_000 == 0 and stats.records_processed > 0:
                stats.log_progress()

    def _parse_csv(self, cache_path: Path, data_year: int) -> list[dict]:
        """Parse TRI CSV file downloaded from Envirofacts.

        Older TRI files (and some cached copies) prefix every column header with a
        sequence number, e.g. "4. FACILITY NAME" instead of "FACILITY NAME".
        Strip that prefix so column lookups work regardless of file vintage.
        """
        try:
            with open(cache_path, encoding="latin-1", errors="replace") as fh:
                text = fh.read()
            # TRI files are comma-separated; detect in case of tab-delimited legacy cache
            delimiter = "\t" if "\t" in text[:2000] else ","
            reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
            rows = list(reader)
            if not rows:
                return rows
            # Detect numbered headers: "1. YEAR", "4. FACILITY NAME", etc.
            sample_key = next(iter(rows[0]))
            if sample_key and sample_key[0].isdigit() and ". " in sample_key:
                import re as _re

                _strip = _re.compile(r"^\d+\.\s+")
                rows = [{_strip.sub("", k): v for k, v in row.items()} for row in rows]
                logger.info("  Stripped numbered column prefixes from year %d headers.", data_year)
            return rows
        except Exception as exc:
            logger.error("Failed to parse CSV for year %d: %s", data_year, exc)
            return []

    async def _process_row(self, row: dict, stats: IngestStats) -> None:
        facility = _col(row, _COL_FACILITY, "FACILITY")
        chemical = _col(row, _COL_CHEMICAL, "CHEMICAL NAME")
        state = _col(row, _COL_STATE, "STATE")
        year_str = _col(row, _COL_YEAR, "REPORTING YEAR")

        if not facility or not chemical:
            stats.records_skipped += 1
            return

        # Use grand total (on-site + off-site) for threshold filtering
        total_all = _float(_col(row, _COL_TOTAL_RELEASES))
        if total_all == 0.0:
            # Fallback: sum on-site + off-site separately
            total_all = _float(_col(row, _COL_ONSITE_TOTAL)) + _float(
                _col(row, _COL_OFFSITE_TOTAL, "TOTAL TRANSFERS OFF-SITE")
            )

        if total_all < self.min_release_lbs:
            stats.records_skipped += 1
            return

        doc, report_unix = build_document(row)
        if not doc.strip():
            stats.records_skipped += 1
            return

        stats.records_processed += 1

        parent = _col(row, _COL_PARENT, "PARENT COMPANY")
        city = _col(row, _COL_CITY)
        county = _col(row, _COL_COUNTY)
        cas = _col(row, _COL_CAS, "CAS #", "CAS NUMBER")
        carcinogen = _col(row, _COL_CARCINOGEN)

        # Stable ID: hash of facility+chemical+year
        record_key = f"epa:tri:{facility}:{chemical}:{year_str}"
        point_id = str(uuid.UUID(bytes=hashlib.sha256(record_key.encode()).digest()[:16]))

        entity_tags = [t for t in [facility, parent, chemical, city, state, county] if t]
        ingest_unix = report_unix or int(time.time())

        # Normalised release weight [0,1]: log1p-scaled, ceiling at 10 million lbs.
        # Used by the orchestrator as a score multiplier for boost-collection queries.
        release_weight = min(1.0, math.log1p(total_all) / math.log1p(10_000_000)) if total_all > 0 else 0.0

        payload: dict = {
            "text": doc,
            "source": SOURCE_LABEL,
            "document_type": "toxic_release",
            "provenance": "epa_tri",
            "ingest_date": TODAY,
            "facility": facility,
            "chemical": chemical,
            "state": state,
            "reporting_year": year_str,
            "entity_tags": entity_tags,
            "ingested_at_unix": ingest_unix,
            "total_releases_lbs": total_all,
            "release_weight": release_weight,
        }
        if parent:
            payload["parent_company"] = parent
        if city:
            payload["city"] = city
        if county:
            payload["county"] = county
        if cas:
            payload["cas_number"] = cas
        if carcinogen and carcinogen.upper() in ("YES", "Y"):
            payload["carcinogen"] = True

        self._upsert_buffer.append(
            qdrant_models.PointStruct(id=point_id, vector=[0.0] * EMBEDDING_DIM, payload=payload)
        )

        if len(self._upsert_buffer) >= self.upsert_batch_size:
            await self._flush_upsert_buffer(stats)

        if self.enqueue_notable and total_all >= NOTABLE_RELEASE_THRESHOLD_LBS:
            await self._maybe_enqueue(facility, parent, chemical, state, year_str, total_all, stats)

    async def _maybe_enqueue(
        self,
        facility: str,
        parent: str,
        chemical: str,
        state: str,
        year: str,
        total_lbs: float,
        stats: IngestStats,
    ) -> None:
        if not self._redis or self.dry_run:
            return
        record_key = f"epa:tri:enqueued:{facility}:{chemical}:{year}"
        redis_key = f"osia:epa_tri:enqueued:{hashlib.md5(record_key.encode()).hexdigest()}"  # noqa: S324
        if await self._redis.exists(redis_key):
            return
        corp_name = parent if parent else facility
        topic = f"EPA TRI: {corp_name} released {total_lbs:,.0f} lbs of {chemical} in {state} ({year})"
        job = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": topic,
                "desk": "environment-and-ecology-desk",
                "priority": "normal",
                "triggered_by": "epa_tri_ingest",
                "metadata": {
                    "facility": facility,
                    "parent_company": parent,
                    "chemical": chemical,
                    "state": state,
                    "year": year,
                },
            }
        )
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        await self._redis.set(redis_key, "1", ex=60 * 60 * 24 * 60)
        stats.facilities_enqueued += 1
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
        description="Ingest EPA Toxics Release Inventory into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="Skip Qdrant writes and Redis updates")
    p.add_argument("--resume", action="store_true", help="Skip years already completed")
    p.add_argument("--year-from", type=int, default=2015, help="First data year to ingest")
    p.add_argument("--year-to", type=int, default=2023, help="Last data year to ingest")
    p.add_argument("--min-release-lbs", type=float, default=1000.0, help="Min total release lbs to include")
    p.add_argument("--enqueue-notable", action="store_true", help="Push top polluters to research queue")
    p.add_argument("--data-dir", default="/tmp/epa_tri", help="Local cache directory")  # noqa: S108
    p.add_argument("--embed-batch-size", type=int, default=48, help="Texts per HF embedding call")
    p.add_argument("--embed-concurrency", type=int, default=3, help="Parallel embedding calls")
    p.add_argument("--upsert-batch-size", type=int, default=64, help="Points per Qdrant upsert call")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    ingestor = EpaTRIIngestor(args)
    asyncio.run(ingestor.run())
