"""
OSIA ICIJ Offshore Leaks Ingestion

Downloads the ICIJ Offshore Leaks full CSV archive (Panama Papers, Pandora Papers,
Paradise Papers, Bahamas Leaks, Offshore Leaks — 810K+ entities), joins entities
with their officers, intermediaries, and addresses via the relationships graph,
builds rich structured documents per entity, and upserts into the
'icij-offshore-leaks' Qdrant collection for Finance desk RAG retrieval.

Datasets covered:
  - Offshore Leaks (2013)
  - Panama Papers (2016)
  - Bahamas Leaks (2016)
  - Paradise Papers (2017)
  - Pandora Papers (2021)

Usage:
  uv run python scripts/ingest_icij_offshore_leaks.py
  uv run python scripts/ingest_icij_offshore_leaks.py --limit 5000 --dry-run
  uv run python scripts/ingest_icij_offshore_leaks.py --resume
  uv run python scripts/ingest_icij_offshore_leaks.py --source-filter "Panama Papers" "Pandora Papers"
  uv run python scripts/ingest_icij_offshore_leaks.py --enqueue-notable

Options:
  --limit N             Stop after N entities (0 = no limit)
  --source-filter       Only ingest entities from these source datasets (space-separated).
                        e.g. "Panama Papers" "Pandora Papers" "Paradise Papers"
                        Default: all sources
  --enqueue-notable     Push entities from high-risk jurisdictions to Finance desk
                        research queue for deeper analysis
  --resume              Resume from last Redis checkpoint
  --dry-run             Build documents but skip Qdrant writes and Redis updates
  --embed-batch-size    Texts per HF embedding call (default: 48)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)
  --data-dir            Directory to cache downloaded CSV files
                        (default: /tmp/icij_offshore_leaks)

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
import json
import logging
import os
import time
import uuid
import zipfile
from collections import defaultdict
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
logger = logging.getLogger("osia.icij_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "icij-offshore-leaks"
EMBEDDING_DIM = 384
SOURCE_LABEL = "ICIJ Offshore Leaks Database"

DOWNLOAD_URL = "https://offshoreleaks-data.icij.org/offshoreleaks/csv/full-oldb.LATEST.zip"
DEFAULT_DATA_DIR = Path("/tmp/icij_offshore_leaks")  # noqa: S108

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

# Redis keys
CHECKPOINT_KEY = "osia:icij:checkpoint"
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# Jurisdictions that warrant automatic research queue enqueuing
HIGH_RISK_JURISDICTIONS = {
    "PAN",  # Panama
    "VGB",  # British Virgin Islands
    "CYM",  # Cayman Islands
    "BMU",  # Bermuda
    "BHS",  # Bahamas
    "SCG",  # Serbia and Montenegro (historical)
    "LIE",  # Liechtenstein
    "MCO",  # Monaco
    "WSM",  # Samoa
    "COK",  # Cook Islands
    "VUT",  # Vanuatu
    "SYC",  # Seychelles
    "MHL",  # Marshall Islands
    "NRU",  # Nauru
    "PLW",  # Palau
}

# CSV filenames expected inside the zip
CSV_ENTITIES = "nodes-entities.csv"
CSV_OFFICERS = "nodes-officers.csv"
CSV_INTERMEDIARIES = "nodes-intermediaries.csv"
CSV_ADDRESSES = "nodes-addresses.csv"
CSV_RELATIONSHIPS = "relationships.csv"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    records_seen: int = 0
    records_skipped: int = 0
    records_processed: int = 0
    chunks_produced: int = 0
    points_upserted: int = 0
    entities_enqueued: int = 0
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
            self.entities_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _find_csv(zf: zipfile.ZipFile, filename: str) -> str | None:
    """Return the full path inside the zip for a given CSV filename."""
    for name in zf.namelist():
        if name.endswith(filename):
            return name
    return None


def download_and_extract(data_dir: Path) -> Path:
    """
    Download the ICIJ full CSV zip if not already cached, then extract.
    Returns the directory containing the extracted CSV files.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "full-oldb.LATEST.zip"

    if not zip_path.exists():
        logger.info("Downloading ICIJ Offshore Leaks CSV archive...")
        logger.info("Source: %s", DOWNLOAD_URL)
        with httpx.Client(timeout=600.0, follow_redirects=True) as http:
            with http.stream("GET", DOWNLOAD_URL) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                last_log = 0
                with open(zip_path, "wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total and downloaded - last_log > 50 * 1024 * 1024:
                            pct = downloaded / total * 100
                            logger.info(
                                "  %.0f%% — %d / %d MB",
                                pct,
                                downloaded // (1024 * 1024),
                                total // (1024 * 1024),
                            )
                            last_log = downloaded
        logger.info("Download complete: %s (%.1f MB)", zip_path, zip_path.stat().st_size / (1024 * 1024))
    else:
        logger.info("Using cached archive: %s", zip_path)

    extract_dir = data_dir / "csv"
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        logger.info("Extracting archive to %s ...", extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        logger.info("Extraction complete.")
    else:
        logger.info("Using cached extraction: %s", extract_dir)

    return extract_dir


# ---------------------------------------------------------------------------
# Graph index builders
# ---------------------------------------------------------------------------


def _safe(val: object, default: str = "") -> str:
    if val is None:
        return default
    s = str(val).strip()
    return s if s not in ("None", "nan", "NaT", "") else default


def build_officer_index(csv_dir: Path) -> dict[str, str]:
    """node_id → officer name"""
    index: dict[str, str] = {}
    path = csv_dir / CSV_OFFICERS
    if not path.exists():
        # Search subdirectory
        matches = list(csv_dir.rglob(CSV_OFFICERS))
        if not matches:
            logger.warning("Officers CSV not found in %s", csv_dir)
            return index
        path = matches[0]
    with open(path, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            nid = _safe(row.get("node_id"))
            name = _safe(row.get("name"))
            if nid and name:
                index[nid] = name
    logger.info("Officer index: %d nodes", len(index))
    return index


def build_intermediary_index(csv_dir: Path) -> dict[str, tuple[str, str]]:
    """node_id → (intermediary name, countries)"""
    index: dict[str, tuple[str, str]] = {}
    path = csv_dir / CSV_INTERMEDIARIES
    if not path.exists():
        matches = list(csv_dir.rglob(CSV_INTERMEDIARIES))
        if not matches:
            logger.warning("Intermediaries CSV not found in %s", csv_dir)
            return index
        path = matches[0]
    with open(path, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            nid = _safe(row.get("node_id"))
            name = _safe(row.get("name"))
            countries = _safe(row.get("countries"))
            if nid and name:
                index[nid] = (name, countries)
    logger.info("Intermediary index: %d nodes", len(index))
    return index


def build_address_index(csv_dir: Path) -> dict[str, str]:
    """node_id → address string"""
    index: dict[str, str] = {}
    path = csv_dir / CSV_ADDRESSES
    if not path.exists():
        matches = list(csv_dir.rglob(CSV_ADDRESSES))
        if not matches:
            logger.warning("Addresses CSV not found in %s", csv_dir)
            return index
        path = matches[0]
    with open(path, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            nid = _safe(row.get("node_id"))
            addr = _safe(row.get("address") or row.get("name"))
            if nid and addr:
                index[nid] = addr
    logger.info("Address index: %d nodes", len(index))
    return index


def build_relationship_index(
    csv_dir: Path,
    officer_names: dict[str, str],
    intermediary_names: dict[str, tuple[str, str]],
    address_index: dict[str, str],
) -> tuple[
    dict[str, list[tuple[str, str]]],  # entity_id → [(role, officer_name), ...]
    dict[str, list[tuple[str, str]]],  # entity_id → [(intermediary_name, countries), ...]
    dict[str, list[str]],  # entity_id → [address_str, ...]
]:
    """
    Parse relationships.csv and build per-entity lookup dicts for officers,
    intermediaries, and addresses.
    """
    entity_officers: dict[str, list[tuple[str, str]]] = defaultdict(list)
    entity_intermediaries: dict[str, list[tuple[str, str]]] = defaultdict(list)
    entity_addresses: dict[str, list[str]] = defaultdict(list)

    path = csv_dir / CSV_RELATIONSHIPS
    if not path.exists():
        matches = list(csv_dir.rglob(CSV_RELATIONSHIPS))
        if not matches:
            logger.warning("Relationships CSV not found in %s", csv_dir)
            return entity_officers, entity_intermediaries, entity_addresses
        path = matches[0]

    total = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            total += 1
            start = _safe(row.get("node_id_start"))
            end = _safe(row.get("node_id_end"))
            link = _safe(row.get("link") or row.get("rel_type"))

            if not start or not end:
                continue

            # Officer → Entity  (start=officer, end=entity)
            if end in officer_names or start in officer_names:
                # The officer can be either side depending on direction
                # Convention: officer node_id_start, entity node_id_end
                if start in officer_names:
                    entity_officers[end].append((link, officer_names[start]))
                elif end in officer_names:
                    entity_officers[start].append((link, officer_names[end]))

            # Intermediary → Entity
            if start in intermediary_names:
                entity_intermediaries[end].append(intermediary_names[start])
            elif end in intermediary_names:
                entity_intermediaries[start].append(intermediary_names[end])

            # Address
            if start in address_index:
                entity_addresses[end].append(address_index[start])
            elif end in address_index:
                entity_addresses[start].append(address_index[end])

    logger.info(
        "Relationships parsed: %d total | %d entities with officers | %d with intermediaries | %d with addresses",
        total,
        len(entity_officers),
        len(entity_intermediaries),
        len(entity_addresses),
    )
    return entity_officers, entity_intermediaries, entity_addresses


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


def build_document(
    row: dict,
    entity_officers: dict[str, list[tuple[str, str]]],
    entity_intermediaries: dict[str, list[tuple[str, str]]],
    entity_addresses: dict[str, list[str]],
) -> str:
    """
    Build a rich structured text document for a single entity, incorporating
    all connected officers, intermediaries, and addresses.
    """
    node_id = _safe(row.get("node_id"))
    name = _safe(row.get("name"))
    original_name = _safe(row.get("original_name"))
    former_name = _safe(row.get("former_name"))
    company_type = _safe(row.get("company_type"))
    jurisdiction = _safe(row.get("jurisdiction"))
    jurisdiction_desc = _safe(row.get("jurisdiction_description"))
    status = _safe(row.get("status"))
    incorporation_date = _safe(row.get("incorporation_date"))
    inactivation_date = _safe(row.get("inactivation_date"))
    struck_off_date = _safe(row.get("struck_off_date"))
    closed_date = _safe(row.get("closed_date"))
    countries = _safe(row.get("countries"))
    service_provider = _safe(row.get("service_provider"))
    source_id = _safe(row.get("sourceID"))
    note = _safe(row.get("note"))
    address_inline = _safe(row.get("address"))

    lines: list[str] = []

    lines.append(f"Source Dataset: {source_id}")
    lines.append(f"Entity: {name}")
    if original_name and original_name != name:
        lines.append(f"Original Name: {original_name}")
    if former_name and former_name != name:
        lines.append(f"Former Name: {former_name}")
    if company_type:
        lines.append(f"Entity Type: {company_type}")

    if jurisdiction_desc:
        jur_str = jurisdiction_desc
        if jurisdiction and jurisdiction != jurisdiction_desc:
            jur_str += f" ({jurisdiction})"
        lines.append(f"Jurisdiction: {jur_str}")
    elif jurisdiction:
        lines.append(f"Jurisdiction: {jurisdiction}")

    if status:
        lines.append(f"Status: {status}")
    if incorporation_date:
        lines.append(f"Incorporated: {incorporation_date}")
    if inactivation_date:
        lines.append(f"Inactivated: {inactivation_date}")
    if struck_off_date:
        lines.append(f"Struck Off: {struck_off_date}")
    if closed_date:
        lines.append(f"Closed: {closed_date}")
    if countries:
        lines.append(f"Connected Countries: {countries}")
    if service_provider:
        lines.append(f"Service Provider: {service_provider}")

    # Officers
    officers = entity_officers.get(node_id, [])
    if officers:
        lines.append("")
        lines.append("Officers / Beneficial Owners:")
        for role, oname in officers[:40]:  # cap at 40 to avoid absurdly long docs
            role_str = f" ({role})" if role else ""
            lines.append(f"  - {oname}{role_str}")
        if len(officers) > 40:
            lines.append(f"  ... and {len(officers) - 40} more")

    # Intermediaries
    intermediaries = entity_intermediaries.get(node_id, [])
    if intermediaries:
        lines.append("")
        lines.append("Intermediaries (law firms / financial agents):")
        for iname, icountries in intermediaries[:10]:
            country_str = f" [{icountries}]" if icountries else ""
            lines.append(f"  - {iname}{country_str}")

    # Addresses
    addresses = entity_addresses.get(node_id, [])
    if not addresses and address_inline:
        addresses = [address_inline]
    if addresses:
        lines.append("")
        lines.append("Registered Address:")
        for addr in addresses[:3]:
            lines.append(f"  {addr}")

    if note:
        lines.append("")
        lines.append(f"Note: {note}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class IcijOffshoreLeaksIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.enqueue_notable: bool = args.enqueue_notable
        self.source_filter: set[str] | None = {s.strip() for s in args.source_filter} if args.source_filter else None
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.data_dir: Path = Path(args.data_dir)

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self, limit: int, resume: bool) -> None:
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await self._ensure_collection()

            checkpoint = 0
            if resume:
                checkpoint = await self._load_checkpoint()
                if checkpoint:
                    logger.info("Resuming from record %d", checkpoint)

            # Download + extract (sync — runs before event loop work begins)
            csv_dir = await asyncio.get_event_loop().run_in_executor(None, download_and_extract, self.data_dir)

            # Build graph indices (sync — CPU-bound, run in executor)
            logger.info("Building graph indices from CSV files...")
            (
                officer_names,
                intermediary_names,
                address_idx,
                rel_indices,
            ) = await asyncio.get_event_loop().run_in_executor(None, self._build_indices, csv_dir)
            entity_officers, entity_intermediaries, entity_addresses = rel_indices

            logger.info(
                "Starting entity ingestion (limit=%s source_filter=%s resume_from=%d)",
                limit or "none",
                self.source_filter or "all",
                checkpoint,
            )

            stats = IngestStats()
            await self._ingest(
                csv_dir,
                stats,
                limit,
                checkpoint,
                entity_officers,
                entity_intermediaries,
                entity_addresses,
            )

            await self._flush_upsert_buffer(stats)
            await self._save_checkpoint(stats.records_seen)
            stats.log_progress()
            logger.info("Ingestion complete.")
        finally:
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()

    def _build_indices(self, csv_dir: Path):
        officer_names = build_officer_index(csv_dir)
        intermediary_names = build_intermediary_index(csv_dir)
        address_idx = build_address_index(csv_dir)
        rel_indices = build_relationship_index(csv_dir, officer_names, intermediary_names, address_idx)
        return officer_names, intermediary_names, address_idx, rel_indices

    # ------------------------------------------------------------------
    # Collection setup
    # ------------------------------------------------------------------

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
                optimizers_config=qdrant_models.OptimizersConfigDiff(
                    indexing_threshold=1000,
                ),
            )
            logger.info("Created Qdrant collection '%s'.", COLLECTION_NAME)
        else:
            info = await self._qdrant.get_collection(COLLECTION_NAME)
            logger.info(
                "Collection '%s' ready (%d points).",
                COLLECTION_NAME,
                info.points_count or 0,
            )

    # ------------------------------------------------------------------
    # Entity ingestion
    # ------------------------------------------------------------------

    async def _ingest(
        self,
        csv_dir: Path,
        stats: IngestStats,
        limit: int,
        checkpoint: int,
        entity_officers: dict,
        entity_intermediaries: dict,
        entity_addresses: dict,
    ) -> None:
        entities_path = csv_dir / CSV_ENTITIES
        if not entities_path.exists():
            matches = list(csv_dir.rglob(CSV_ENTITIES))
            if not matches:
                raise FileNotFoundError(f"{CSV_ENTITIES} not found in {csv_dir}")
            entities_path = matches[0]

        loop = asyncio.get_event_loop()

        def _iter_entities():
            with open(entities_path, encoding="utf-8", errors="replace") as fh:
                yield from csv.DictReader(fh)

        it = iter(_iter_entities())
        _sentinel = object()

        while True:
            row = await loop.run_in_executor(None, next, it, _sentinel)
            if row is _sentinel:
                logger.info("Entities CSV exhausted at %d records.", stats.records_seen)
                break

            stats.records_seen += 1

            if stats.records_seen <= checkpoint:
                if stats.records_seen % 50_000 == 0:
                    logger.info("Fast-forwarding checkpoint: %d/%d", stats.records_seen, checkpoint)
                continue

            source_id = _safe(row.get("sourceID"))
            if self.source_filter and source_id not in self.source_filter:
                stats.records_skipped += 1
                continue

            name = _safe(row.get("name"))
            if not name:
                stats.records_skipped += 1
                continue

            try:
                await self._process_entity(row, stats, entity_officers, entity_intermediaries, entity_addresses)
            except Exception as exc:
                stats.errors += 1
                logger.warning("Error processing entity %s: %s", name, exc)

            if stats.records_processed % 5000 == 0 and stats.records_processed > 0:
                stats.log_progress()

            if limit and stats.records_seen - checkpoint >= limit:
                logger.info("Reached --limit %d — stopping.", limit)
                break

        await self._save_checkpoint(stats.records_seen)

    async def _process_entity(
        self,
        row: dict,
        stats: IngestStats,
        entity_officers: dict,
        entity_intermediaries: dict,
        entity_addresses: dict,
    ) -> None:
        stats.records_processed += 1

        node_id = _safe(row.get("node_id"))
        name = _safe(row.get("name"))
        source_id = _safe(row.get("sourceID"))
        jurisdiction = _safe(row.get("jurisdiction"))
        countries = _safe(row.get("countries"))
        incorporation_date = _safe(row.get("incorporation_date"))
        status = _safe(row.get("status"))
        company_type = _safe(row.get("company_type"))

        doc = build_document(row, entity_officers, entity_intermediaries, entity_addresses)
        if not doc.strip():
            stats.records_skipped += 1
            return

        stats.chunks_produced += 1

        point_id = str(uuid.UUID(bytes=hashlib.sha256(f"icij:{node_id}".encode()).digest()[:16]))

        officers = entity_officers.get(node_id, [])
        officer_names_list = [oname for _, oname in officers[:20]]
        intermediaries = entity_intermediaries.get(node_id, [])
        intermediary_names_list = [iname for iname, _ in intermediaries[:5]]

        entity_tags = [t for t in [name, source_id, jurisdiction, countries] if t]
        entity_tags += officer_names_list[:5]

        payload: dict = {
            "text": doc,
            "source": SOURCE_LABEL,
            "document_type": "offshore_entity",
            "provenance": "icij_offshore_leaks",
            "ingest_date": TODAY,
            "node_id": node_id,
            "entity_name": name,
            "source_dataset": source_id,
            "entity_tags": entity_tags,
            "ingested_at_unix": int(time.time()),
        }
        if jurisdiction:
            payload["jurisdiction"] = jurisdiction
        if countries:
            payload["countries"] = countries
        if status:
            payload["status"] = status
        if incorporation_date:
            payload["incorporation_date"] = incorporation_date
        if company_type:
            payload["company_type"] = company_type
        if officer_names_list:
            payload["officers"] = officer_names_list
        if intermediary_names_list:
            payload["intermediaries"] = intermediary_names_list

        self._upsert_buffer.append(
            qdrant_models.PointStruct(
                id=point_id,
                vector=[0.0] * EMBEDDING_DIM,
                payload=payload,
            )
        )

        if len(self._upsert_buffer) >= self.upsert_batch_size:
            await self._flush_upsert_buffer(stats)

        # Enqueue notable entities for Finance desk deep research
        if self.enqueue_notable and jurisdiction in HIGH_RISK_JURISDICTIONS:
            await self._maybe_enqueue_entity(node_id, name, source_id, jurisdiction, stats)

    # ------------------------------------------------------------------
    # Research queue
    # ------------------------------------------------------------------

    async def _maybe_enqueue_entity(
        self,
        node_id: str,
        name: str,
        source_id: str,
        jurisdiction: str,
        stats: IngestStats,
    ) -> None:
        if not self._redis or self.dry_run:
            return

        redis_key = f"osia:icij:enqueued:{node_id}"
        if await self._redis.exists(redis_key):
            return

        job = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": f"ICIJ {source_id} offshore entity: {name} (jurisdiction: {jurisdiction})",
                "desk": "finance-and-economics-directorate",
                "priority": "low",
                "triggered_by": "icij_offshore_leaks_ingest",
                "metadata": {
                    "node_id": node_id,
                    "entity_name": name,
                    "source_dataset": source_id,
                    "jurisdiction": jurisdiction,
                },
            }
        )
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        await self._redis.set(redis_key, "1", ex=60 * 60 * 24 * 30)
        stats.entities_enqueued += 1
        logger.debug("Enqueued Finance desk research: %r (%s)", name, jurisdiction)

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
            await self._qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )
            logger.debug("Upserted %d points to '%s'.", len(points), COLLECTION_NAME)

        if stats is not None:
            stats.points_upserted += len(points)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    async def _save_checkpoint(self, cursor: int) -> None:
        if self.dry_run or not self._redis:
            return
        await self._redis.set(CHECKPOINT_KEY, cursor)
        logger.info("Checkpoint saved: %d records", cursor)

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
        description="Ingest ICIJ Offshore Leaks database into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--limit", type=int, default=0, help="Stop after N entities (0=no limit)")
    p.add_argument(
        "--source-filter",
        nargs="+",
        metavar="DATASET",
        dest="source_filter",
        help='Only ingest entities from these source datasets, e.g. "Panama Papers" "Pandora Papers"',
    )
    p.add_argument(
        "--enqueue-notable",
        action="store_true",
        dest="enqueue_notable",
        help="Push entities from high-risk jurisdictions to Finance desk research queue",
    )
    p.add_argument("--resume", action="store_true", help="Resume from Redis checkpoint")
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Build documents but skip Qdrant writes and Redis updates",
    )
    p.add_argument("--embed-batch-size", type=int, default=48, dest="embed_batch_size")
    p.add_argument("--embed-concurrency", type=int, default=3, dest="embed_concurrency")
    p.add_argument("--upsert-batch-size", type=int, default=64, dest="upsert_batch_size")
    p.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        dest="data_dir",
        help="Directory for cached CSV download and extraction",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not HF_TOKEN:
        parser.error("HF_TOKEN not set in environment — required for embeddings.")

    logger.info(
        "ICIJ Offshore Leaks ingest | limit=%s source_filter=%s enqueue_notable=%s dry_run=%s data_dir=%s",
        args.limit or "none",
        args.source_filter or "all",
        args.enqueue_notable,
        args.dry_run,
        args.data_dir,
    )

    if args.dry_run:
        logger.warning("DRY RUN — no data will be written to Qdrant or Redis.")

    ingestor = IcijOffshoreLeaksIngestor(args)
    asyncio.run(ingestor.run(limit=args.limit, resume=args.resume))


if __name__ == "__main__":
    main()
