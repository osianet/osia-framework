"""
OSIA Iran-Israel War 2026 Dataset Ingestion

Ingests the danielrosehill/Iran-Israel-War-2026 HuggingFace dataset into a
dedicated 'iran-israel-war-2026' Qdrant collection. Covers three tables:

  waves               — 53+ attack wave records (89 structured fields)
  international_reactions — 210+ country/org reactions (33 fields)
  incidents           — incident-level records

Delta detection: the last-ingested HuggingFace commit SHA is stored in Redis.
On each run the current SHA is fetched from the HF Hub API; if it matches the
stored value the script exits early (unless --force is passed). Qdrant upserts
are idempotent via stable deterministic point IDs, so safe to re-run.

Dataset: danielrosehill/Iran-Israel-War-2026
  - Tracks Iranian missile/drone attack waves (True Promise 1–4, 2024–2026)
  - Weapons, targets, interception, casualties, international reactions
  - License: CC-BY-4.0

Usage:
  uv run python scripts/ingest_iran_israel_war.py
  uv run python scripts/ingest_iran_israel_war.py --dry-run
  uv run python scripts/ingest_iran_israel_war.py --force
  uv run python scripts/ingest_iran_israel_war.py --tables waves reactions
  uv run python scripts/ingest_iran_israel_war.py --enqueue-research

Options:
  --dry-run        Parse and build documents but skip Qdrant writes and Redis updates
  --force          Re-ingest even if the commit SHA has not changed
  --tables         Which tables to ingest: waves, reactions, incidents (default: all)
  --enqueue-research  Push each wave to Geopolitical desk research queue for deeper analysis
  --embed-batch-size  Texts per HF embedding call (default: 32)
  --embed-concurrency Parallel embedding calls (default: 3)
  --upsert-batch-size Points per Qdrant upsert call (default: 64)

Environment variables (from .env):
  HF_TOKEN          HuggingFace token (required for dataset access + embeddings)
  QDRANT_URL        Qdrant URL (default: https://qdrant.osia.dev)
  QDRANT_API_KEY    Qdrant API key
  REDIS_URL         Redis URL (default: redis://localhost:6379)
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
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
logger = logging.getLogger("osia.iran_israel_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "iran-israel-war-2026"
EMBEDDING_DIM = 384

HF_DATASET_ID = "danielrosehill/Iran-Israel-War-2026"
SOURCE_LABEL = "Iran-Israel War 2026 OSINT Dataset (danielrosehill/Iran-Israel-War-2026)"

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)
HF_API_URL = f"https://huggingface.co/api/datasets/{HF_DATASET_ID}"

# Redis keys
COMMIT_SHA_KEY = "osia:iran_israel_war:last_commit"
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

VALID_TABLES = {"waves", "reactions", "incidents"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(val, default: str = "") -> str:
    if val is None:
        return default
    s = str(val).strip()
    return s if s not in ("None", "nan", "NaT", "") else default


def _bool_flag(val) -> bool | None:
    """Convert numpy bool / Python bool / string flag to Python bool."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _stable_id(namespace: str, key: str) -> str:
    """Deterministic UUID from namespace + key, stable across runs."""
    digest = hashlib.sha256(f"{namespace}:{key}".encode()).digest()[:16]
    return str(uuid.UUID(bytes=digest))


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------


def build_wave_document(row: dict) -> str:
    """
    Build a rich narrative text document from a waves table row.
    Includes all operationally meaningful fields in a readable format.
    """
    lines: list[str] = []

    operation = _safe(row.get("operation"))
    wave_num = _safe(row.get("wave_number"))
    wave_uid = _safe(row.get("wave_uid"))
    codename_en = _safe(row.get("wave_codename_english"))
    codename_fa = _safe(row.get("wave_codename_farsi"))

    header = f"Operation: {operation}" if operation else "Operation: Unknown"
    if wave_num:
        header += f" — Wave {wave_num}"
    if codename_en:
        header += f" ({codename_en}"
        if codename_fa:
            header += f" / {codename_fa}"
        header += ")"
    lines.append(header)
    if wave_uid:
        lines.append(f"Wave UID: {wave_uid}")

    # Timing
    timing_parts = []
    for field_name, label in [
        ("announced_utc", "Announced"),
        ("probable_launch_time", "Probable Launch"),
        ("launch_time_israel", "Launch (Israel TZ)"),
        ("conflict_day", "Conflict Day"),
        ("wave_duration_minutes", "Wave Duration (min)"),
        ("hours_since_last_wave", "Hours Since Last Wave"),
    ]:
        val = _safe(row.get(field_name))
        if val:
            timing_parts.append(f"{label}: {val}")
    if timing_parts:
        lines.append("Timing: " + " | ".join(timing_parts))

    # Weapons
    weapon_cols = [
        ("drones_used", "Drones"),
        ("ballistic_missiles_used", "Ballistic Missiles"),
        ("cruise_missiles_used", "Cruise Missiles"),
        ("emad_used", "Emad MRBM"),
        ("ghadr_used", "Ghadr MRBM"),
        ("sejjil_used", "Sejjil"),
        ("kheibar_shekan_used", "Kheibar Shekan"),
        ("fattah_used", "Fattah Hypersonic"),
        ("shahed_136_used", "Shahed-136"),
        ("shahed_238_used", "Shahed-238"),
        ("shahed_131_used", "Shahed-131"),
        ("shahed_107_used", "Shahed-107"),
        ("shahed_129_used", "Shahed-129"),
        ("mohajer_6_used", "Mohajer-6"),
        ("bm_marv_equipped", "MARV-equipped BMs"),
        ("bm_hypersonic", "Hypersonic BMs"),
        ("bm_cluster_warhead", "Cluster Warhead BMs"),
    ]
    active_weapons = [label for col, label in weapon_cols if _bool_flag(row.get(col)) is True]
    if active_weapons:
        lines.append("Weapons Employed: " + ", ".join(active_weapons))

    # Munitions count
    munitions = _safe(row.get("estimated_munitions_count"))
    if munitions:
        parts = [f"Total ~{munitions}"]
        t_israel = _safe(row.get("munitions_targeting_israel"))
        t_us = _safe(row.get("munitions_targeting_us_bases"))
        if t_israel:
            parts.append(f"targeting Israel: {t_israel}")
        if t_us:
            parts.append(f"targeting US bases: {t_us}")
        cumulative = _safe(row.get("cumulative_total"))
        if cumulative:
            parts.append(f"cumulative campaign total: {cumulative}")
        lines.append("Munitions: " + " | ".join(parts))

    # Targets
    target_text = _safe(row.get("targets"))
    if target_text:
        lines.append(f"Target Description: {target_text}")

    target_flags = [
        ("israel_targeted", "Israel"),
        ("us_bases_targeted", "US Bases"),
        ("target_iaf_base", "IAF Airbase"),
        ("target_us_base", "US Military Base"),
        ("target_naval_base", "Naval Base"),
        ("target_naval_vessel", "Naval Vessel"),
        ("target_government_c2", "Government/C2"),
        ("target_military_industrial", "Military-Industrial"),
        ("target_intelligence", "Intelligence Facility"),
        ("target_civilian_infrastructure", "Civilian Infrastructure"),
        ("target_civilian_area", "Civilian Area"),
        ("target_diplomatic", "Diplomatic"),
        ("targeted_tel_aviv", "Tel Aviv"),
        ("targeted_jerusalem", "Jerusalem"),
        ("targeted_haifa", "Haifa"),
        ("targeted_negev_beersheba", "Negev/Beer-Sheba"),
        ("targeted_northern_periphery", "Northern Periphery"),
        ("targeted_eilat", "Eilat"),
    ]
    active_targets = [label for col, label in target_flags if _bool_flag(row.get(col)) is True]
    if active_targets:
        lines.append("Target Types/Areas: " + ", ".join(active_targets))

    lat = _safe(row.get("target_lat"))
    lon = _safe(row.get("target_lon"))
    if lat and lon:
        lines.append(f"Target Coordinates: {lat}, {lon}")

    # Interception
    intercept_parts = []
    intercept_rate = _safe(row.get("estimated_intercept_rate"))
    if intercept_rate:
        intercept_parts.append(f"Rate: {intercept_rate}")
    intercept_count = _safe(row.get("estimated_intercept_count"))
    if intercept_count:
        intercept_parts.append(f"Count: {intercept_count}")
    systems = _safe(row.get("interception_systems"))
    if systems:
        intercept_parts.append(f"Systems: {systems}")

    interceptors = [
        ("intercepted_by_israel", "Israel"),
        ("intercepted_by_us", "US"),
        ("intercepted_by_uk", "UK"),
        ("intercepted_by_jordan", "Jordan"),
        ("intercepted_by_other", "Other"),
    ]
    active_interceptors = [label for col, label in interceptors if _bool_flag(row.get(col)) is True]
    if active_interceptors:
        intercept_parts.append("By: " + ", ".join(active_interceptors))

    exo = _bool_flag(row.get("exoatmospheric_interception"))
    endo = _bool_flag(row.get("endoatmospheric_interception"))
    if exo:
        intercept_parts.append("Exoatmospheric")
    if endo:
        intercept_parts.append("Endoatmospheric")

    if intercept_parts:
        lines.append("Interception: " + " | ".join(intercept_parts))

    # Impact / casualties
    damage = _safe(row.get("damage"))
    fatalities = _safe(row.get("fatalities"))
    injuries = _safe(row.get("injuries"))
    civilian_cas = _safe(row.get("civilian_casualties"))
    military_cas = _safe(row.get("military_casualties"))
    if damage:
        lines.append(f"Damage Assessment: {damage}")
    if any([fatalities, injuries, civilian_cas, military_cas]):
        cas_parts = []
        if fatalities:
            cas_parts.append(f"Fatalities: {fatalities}")
        if injuries:
            cas_parts.append(f"Injuries: {injuries}")
        if civilian_cas:
            cas_parts.append(f"Civilian casualties: {civilian_cas}")
        if military_cas:
            cas_parts.append(f"Military casualties: {military_cas}")
        lines.append("Casualties: " + " | ".join(cas_parts))

    # Escalation indicators
    escalation_parts = []
    new_country = _safe(row.get("new_country_targeted"))
    if new_country:
        escalation_parts.append(f"New country targeted: {new_country}")
    first_use = _safe(row.get("new_weapon_first_use"))
    if first_use:
        escalation_parts.append(f"First-use weapon: {first_use}")
    proxy = _bool_flag(row.get("proxy_involvement"))
    proxy_desc = _safe(row.get("proxy_description"))
    if proxy:
        escalation_parts.append(f"Proxy involvement: {proxy_desc or 'Yes'}")
    if escalation_parts:
        lines.append("Escalation Indicators: " + " | ".join(escalation_parts))

    # Official statements and narrative
    idf_stmt = _safe(row.get("idf_statement"))
    iran_claim = _safe(row.get("iranian_media_claims"))
    if idf_stmt:
        lines.append(f"IDF Statement: {idf_stmt}")
    if iran_claim:
        lines.append(f"Iranian Media Claims: {iran_claim}")

    description = _safe(row.get("description"))
    narrative = _safe(row.get("narrative"))
    if description:
        lines.append(f"\nDescription: {description}")
    if narrative:
        lines.append(f"\nNarrative: {narrative}")

    sources = _safe(row.get("source_urls"))
    if sources:
        lines.append(f"Sources: {sources}")

    return "\n".join(lines)


def build_reaction_document(row: dict) -> str:
    """Build a narrative document from an international_reactions row."""
    lines: list[str] = []

    entity = _safe(row.get("entity_name"))
    entity_type = _safe(row.get("entity_type"))
    iso = _safe(row.get("iso_3166_1_alpha2"))
    stance = _safe(row.get("overall_stance"))
    combatant = _bool_flag(row.get("combatant"))
    eu_member = _bool_flag(row.get("eu_member_state"))

    header = entity or "Unknown Entity"
    if entity_type:
        header += f" ({entity_type}"
        if iso:
            header += f", {iso}"
        header += ")"
    lines.append(f"Entity: {header}")

    qualifiers = []
    if stance:
        qualifiers.append(f"Overall Stance: {stance}")
    if combatant:
        qualifiers.append("Active Combatant")
    if eu_member:
        qualifiers.append("EU Member State")
    if qualifiers:
        lines.append(" | ".join(qualifiers))

    # Statements at three levels
    for level, speaker_field, date_field, summary_field, text_field, category_field, _url_field in [
        (
            "Head of State",
            "hos_speaker",
            "hos_date",
            "hos_summary",
            "hos_statement_text",
            "hos_statement_category",
            "hos_source_url",
        ),
        (
            "Head of Government",
            "hog_speaker",
            "hog_date",
            "hog_summary",
            "hog_statement_text",
            "hog_statement_category",
            "hog_source_url",
        ),
        (
            "Foreign Ministry",
            "fm_speaker",
            "fm_date",
            "fm_summary",
            "fm_statement_text",
            "fm_statement_category",
            "fm_source_url",
        ),
    ]:
        summary = _safe(row.get(summary_field))
        text = _safe(row.get(text_field))
        speaker = _safe(row.get(speaker_field))
        date = _safe(row.get(date_field))
        category = _safe(row.get(category_field))

        if not summary and not text:
            continue

        stmt_header = f"\n{level}"
        if speaker:
            stmt_header += f" ({speaker}"
            if date:
                stmt_header += f", {date}"
            stmt_header += ")"
        if category:
            stmt_header += f" [{category}]"
        lines.append(stmt_header + ":")

        if summary:
            lines.append(f"  Summary: {summary}")
        if text and text != summary:
            lines.append(f"  Statement: {text}")

    return "\n".join(lines)


def build_incident_document(row: dict) -> str:
    """Build a narrative document from an incidents row."""
    lines: list[str] = []

    incident_id = _safe(row.get("incident_id") or row.get("id"))
    wave_uid = _safe(row.get("wave_uid"))
    title = _safe(row.get("title") or row.get("incident_type"))
    description = _safe(row.get("description"))

    if title:
        lines.append(f"Incident: {title}")
    if incident_id:
        lines.append(f"ID: {incident_id}")
    if wave_uid:
        lines.append(f"Wave: {wave_uid}")

    # Dump all non-empty string fields not already covered
    skip = {"incident_id", "id", "title", "incident_type", "description", "wave_uid"}
    for k, v in row.items():
        if k in skip:
            continue
        val = _safe(v)
        if val:
            label = k.replace("_", " ").title()
            lines.append(f"{label}: {val}")

    if description:
        lines.append(f"\n{description}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    waves_processed: int = 0
    reactions_processed: int = 0
    incidents_processed: int = 0
    points_upserted: int = 0
    research_enqueued: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> str:
        secs = int(time.time() - self.started_at)
        return f"{secs // 60}m{secs % 60:02d}s"

    def log_summary(self) -> None:
        logger.info(
            "waves=%d reactions=%d incidents=%d upserted=%d research_queued=%d errors=%d elapsed=%s",
            self.waves_processed,
            self.reactions_processed,
            self.incidents_processed,
            self.points_upserted,
            self.research_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class IranIsraelWarIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.force: bool = args.force
        self.tables: set[str] = set(args.tables) if args.tables else VALID_TABLES
        self.enqueue_research: bool = args.enqueue_research
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            # Delta check: compare current HF commit SHA to last stored
            current_sha = await self._fetch_current_commit_sha()
            stored_sha = await self._load_stored_sha()

            if current_sha and stored_sha and current_sha == stored_sha and not self.force:
                logger.info(
                    "Dataset is up to date (commit %s). Nothing to ingest. Use --force to override.",
                    current_sha[:12],
                )
                return

            if current_sha and stored_sha and current_sha != stored_sha:
                logger.info(
                    "New dataset commit detected: %s → %s. Ingesting updated data.",
                    stored_sha[:12] if stored_sha else "none",
                    current_sha[:12],
                )
            elif not stored_sha:
                logger.info("No prior ingest found. Starting fresh ingestion.")
            else:
                logger.info("--force passed. Re-ingesting full dataset.")

            await self._ensure_collection()

            stats = IngestStats()
            await self._ingest_all(stats)

            await self._flush_upsert_buffer(stats)

            if current_sha and not self.dry_run:
                await self._save_sha(current_sha)

            stats.log_summary()
            logger.info("Ingestion complete.")
        finally:
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()

    # ------------------------------------------------------------------
    # HF commit SHA detection
    # ------------------------------------------------------------------

    async def _fetch_current_commit_sha(self) -> str:
        """Fetch the latest commit SHA for the dataset from the HF Hub API."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as http:
                headers = {}
                if HF_TOKEN:
                    headers["Authorization"] = f"Bearer {HF_TOKEN}"
                resp = await http.get(HF_API_URL, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                # The top-level sha field is the latest commit
                sha = data.get("sha") or data.get("lastModified") or ""
                if sha:
                    logger.info("Current dataset commit SHA: %s", sha[:12] if len(sha) > 12 else sha)
                return sha
        except Exception as exc:
            logger.warning("Could not fetch dataset SHA from HF API: %s — proceeding with ingest.", exc)
            return ""

    async def _load_stored_sha(self) -> str:
        if not self._redis:
            return ""
        val = await self._redis.get(COMMIT_SHA_KEY)
        return val or ""

    async def _save_sha(self, sha: str) -> None:
        if self.dry_run or not self._redis or not sha:
            return
        await self._redis.set(COMMIT_SHA_KEY, sha)
        logger.info("Stored commit SHA: %s", sha[:12] if len(sha) > 12 else sha)

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
                    indexing_threshold=100,
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
    # Ingestion orchestrator
    # ------------------------------------------------------------------

    async def _load_parquet(self, filename: str) -> list[dict] | None:
        """
        Load a single parquet file from the HF dataset repo, returning rows as dicts.
        Uses pandas + huggingface_hub filesystem to avoid the datasets library
        auto-discovering all CSV files (including data_dictionary.csv) and failing
        on mismatched column schemas.
        """
        import pandas as pd  # type: ignore[import-untyped]

        url = f"hf://datasets/{HF_DATASET_ID}/{filename}"
        storage_opts: dict = {}
        if HF_TOKEN:
            storage_opts["token"] = HF_TOKEN
        try:
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pd.read_parquet(url, storage_options=storage_opts),
            )
            logger.info("Loaded %s: %d rows, %d columns", filename, len(df), len(df.columns))
            # Convert NaN/NaT to None so _safe() handles them uniformly
            return df.where(df.notna(), other=None).to_dict(orient="records")  # type: ignore[return-value]
        except Exception as exc:
            logger.warning("Could not load %s: %s", filename, exc)
            return None

    async def _ingest_all(self, stats: IngestStats) -> None:
        logger.info("Loading tables from %s...", HF_DATASET_ID)

        if "waves" in self.tables:
            rows = await self._load_parquet("waves.parquet")
            if rows:
                logger.info("Ingesting waves table (%d rows)...", len(rows))
                await self._ingest_waves(rows, stats)
            else:
                logger.warning("waves.parquet unavailable — skipping.")

        if "reactions" in self.tables:
            rows = await self._load_parquet("international_reactions.parquet")
            if rows:
                logger.info("Ingesting international reactions table (%d rows)...", len(rows))
                await self._ingest_reactions(rows, stats)
            else:
                logger.warning("international_reactions.parquet unavailable — skipping.")

        if "incidents" in self.tables:
            rows = await self._load_parquet("incidents.parquet")
            if rows:
                logger.info("Ingesting incidents table (%d rows)...", len(rows))
                await self._ingest_incidents(rows, stats)
            else:
                logger.info("incidents.parquet not found — skipping.")

    # ------------------------------------------------------------------
    # Table-specific ingestors
    # ------------------------------------------------------------------

    async def _ingest_waves(self, split, stats: IngestStats) -> None:
        for row in split:
            row = dict(row)
            wave_uid = _safe(row.get("wave_uid"))
            if not wave_uid:
                # Fall back to operation + wave_number
                op = _safe(row.get("operation", "op"))
                wn = _safe(row.get("wave_number", "0"))
                wave_uid = f"{op}_{wn}"

            text = build_wave_document(row)
            if not text.strip():
                continue

            point_id = _stable_id("isr_wave", wave_uid)
            operation = _safe(row.get("operation"))

            payload: dict = {
                "text": text,
                "source": SOURCE_LABEL,
                "dataset": HF_DATASET_ID,
                "document_type": "attack_wave",
                "ingest_date": TODAY,
                "wave_uid": wave_uid,
            }
            if operation:
                payload["operation"] = operation
            wave_num = _safe(row.get("wave_number"))
            if wave_num:
                payload["wave_number"] = wave_num
            announced = _safe(row.get("announced_utc"))
            if announced:
                payload["announced_utc"] = announced[:10] if len(announced) >= 10 else announced
            munitions = _safe(row.get("estimated_munitions_count"))
            if munitions:
                payload["estimated_munitions_count"] = munitions
            fatalities = _safe(row.get("fatalities"))
            if fatalities:
                payload["fatalities"] = fatalities

            self._upsert_buffer.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=[0.0] * EMBEDDING_DIM,
                    payload=payload,
                )
            )
            stats.waves_processed += 1

            if len(self._upsert_buffer) >= self.upsert_batch_size:
                await self._flush_upsert_buffer(stats)

            if self.enqueue_research and not self.dry_run:
                await self._enqueue_wave_research(wave_uid, operation, row)
                stats.research_enqueued += 1

        logger.info("Waves table: %d records processed.", stats.waves_processed)

    async def _ingest_reactions(self, split, stats: IngestStats) -> None:
        for row in split:
            row = dict(row)
            entity = _safe(row.get("entity_name", "unknown"))
            iso = _safe(row.get("iso_3166_1_alpha2", ""))
            key = f"{entity}_{iso}" if iso else entity

            text = build_reaction_document(row)
            if not text.strip():
                continue

            point_id = _stable_id("isr_reaction", key)
            entity_type = _safe(row.get("entity_type"))
            stance = _safe(row.get("overall_stance"))

            payload: dict = {
                "text": text,
                "source": SOURCE_LABEL,
                "dataset": HF_DATASET_ID,
                "document_type": "international_reaction",
                "ingest_date": TODAY,
                "entity_name": entity,
            }
            if iso:
                payload["iso_3166_1_alpha2"] = iso
            if entity_type:
                payload["entity_type"] = entity_type
            if stance:
                payload["overall_stance"] = stance
            combatant = _bool_flag(row.get("combatant"))
            if combatant is not None:
                payload["combatant"] = combatant

            self._upsert_buffer.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=[0.0] * EMBEDDING_DIM,
                    payload=payload,
                )
            )
            stats.reactions_processed += 1

            if len(self._upsert_buffer) >= self.upsert_batch_size:
                await self._flush_upsert_buffer(stats)

        logger.info("Reactions table: %d records processed.", stats.reactions_processed)

    async def _ingest_incidents(self, split, stats: IngestStats) -> None:
        for row in split:
            row = dict(row)
            incident_id = _safe(row.get("incident_id") or row.get("id") or "")
            if not incident_id:
                incident_id = f"incident_{stats.incidents_processed}"

            text = build_incident_document(row)
            if not text.strip():
                continue

            point_id = _stable_id("isr_incident", incident_id)

            payload: dict = {
                "text": text,
                "source": SOURCE_LABEL,
                "dataset": HF_DATASET_ID,
                "document_type": "incident",
                "ingest_date": TODAY,
                "incident_id": incident_id,
            }
            wave_uid = _safe(row.get("wave_uid"))
            if wave_uid:
                payload["wave_uid"] = wave_uid

            self._upsert_buffer.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=[0.0] * EMBEDDING_DIM,
                    payload=payload,
                )
            )
            stats.incidents_processed += 1

            if len(self._upsert_buffer) >= self.upsert_batch_size:
                await self._flush_upsert_buffer(stats)

        logger.info("Incidents table: %d records processed.", stats.incidents_processed)

    # ------------------------------------------------------------------
    # Research queue
    # ------------------------------------------------------------------

    async def _enqueue_wave_research(self, wave_uid: str, operation: str, row: dict) -> None:
        if not self._redis:
            return

        redis_key = f"osia:iran_israel_war:enqueued:{wave_uid}"
        already_queued = await self._redis.exists(redis_key)
        if already_queued:
            return

        topic = f"Iran-Israel War 2026 attack wave: {wave_uid}"
        if operation:
            topic = f"{operation} — {wave_uid}"

        weapons = _safe(row.get("estimated_munitions_count"))
        targets = _safe(row.get("targets"))
        if weapons:
            topic += f" ({weapons} munitions)"
        if targets:
            topic += f" targeting {targets[:80]}"

        job = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": topic,
                "desk": "geopolitical-and-security-desk",
                "priority": "normal",
                "directives_lens": True,
                "triggered_by": "iran_israel_war_ingest",
                "metadata": {
                    "wave_uid": wave_uid,
                    "operation": operation,
                    "dataset": HF_DATASET_ID,
                },
            }
        )
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        # TTL 7 days — avoid re-queuing same wave on next incremental run
        await self._redis.set(redis_key, "1", ex=60 * 60 * 24 * 7)
        logger.debug("Research job enqueued: %r → Geopolitical desk", topic)

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest danielrosehill/Iran-Israel-War-2026 dataset into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and build documents but skip Qdrant writes and Redis updates",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if the HuggingFace commit SHA has not changed",
    )
    p.add_argument(
        "--tables",
        nargs="+",
        metavar="TABLE",
        choices=list(VALID_TABLES),
        help="Which tables to ingest: waves, reactions, incidents (default: all)",
    )
    p.add_argument(
        "--enqueue-research",
        action="store_true",
        dest="enqueue_research",
        help="Push each attack wave to Geopolitical desk research queue for deeper analysis",
    )
    p.add_argument("--embed-batch-size", type=int, default=32, dest="embed_batch_size")
    p.add_argument("--embed-concurrency", type=int, default=3, dest="embed_concurrency")
    p.add_argument("--upsert-batch-size", type=int, default=64, dest="upsert_batch_size")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not HF_TOKEN:
        parser.error("HF_TOKEN not set in environment — required for dataset access and embeddings.")

    logger.info(
        "Iran-Israel War 2026 ingest | tables=%s force=%s dry_run=%s enqueue_research=%s",
        args.tables or list(VALID_TABLES),
        args.force,
        args.dry_run,
        args.enqueue_research,
    )

    if args.dry_run:
        logger.warning("DRY RUN — no data will be written to Qdrant or Redis.")

    ingestor = IranIsraelWarIngestor(args)
    asyncio.run(ingestor.run())


if __name__ == "__main__":
    main()
