"""
OSIA Cybersecurity Attacks Ingestion

Streams the vinitvek/cybersecurityattacks HuggingFace dataset (13,407 documented
incidents), constructs rich incident documents from structured fields, embeds via
the HF Inference API, and upserts into a dedicated 'cybersecurity-attacks' Qdrant
collection.  Threat actor names are optionally enqueued to the Cyber desk research
queue for automated follow-up investigation.

Dataset: vinitvek/cybersecurityattacks
  - 13,407 rows, ~6 MB CSV
  - Covers incidents from 163 countries across 22 industry sectors
  - Fields: slug, event_date, event_year, affected_country, affected_organization,
            affected_industry, event_type, event_subtype, motive, description,
            actor, actor_type, actor_country, source_url

Usage:
  uv run python scripts/ingest_cybersecurity_attacks.py
  uv run python scripts/ingest_cybersecurity_attacks.py --limit 500 --dry-run
  uv run python scripts/ingest_cybersecurity_attacks.py --skip-actors
  uv run python scripts/ingest_cybersecurity_attacks.py --resume

Options:
  --limit N             Stop after N source records (0 = no limit)
  --skip-actors         Disable threat actor research job enqueueing
  --resume              Resume from last checkpoint stored in Redis
  --dry-run             Parse and chunk but skip Qdrant writes and Redis updates
  --embed-batch-size    Texts per HF embedding call (default: 48)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)
  --min-text-len        Minimum chars for a record to be processed (default: 40)

Environment variables (from .env):
  HF_TOKEN              HuggingFace token (required — for dataset access + embeddings)
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
logger = logging.getLogger("osia.cybersec_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "cybersecurity-attacks"
EMBEDDING_DIM = 384

HF_DATASET_ID = "vinitvek/cybersecurityattacks"
SOURCE_LABEL = "Global Cybersecurity Incidents (vinitvek/cybersecurityattacks)"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

# Redis keys
CHECKPOINT_KEY = "osia:cybersec:checkpoint"
SEEN_ACTORS_KEY = "osia:cybersec:seen_actors"  # session-local dedup set
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# Threat actor types that are meaningful research targets
RESEARCHED_ACTOR_TYPES = {"Nation-State", "Hacktivist", "Criminal", "Terrorist"}

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    records_seen: int = 0
    records_skipped: int = 0
    records_processed: int = 0
    points_upserted: int = 0
    actors_enqueued: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> str:
        secs = int(time.time() - self.started_at)
        return f"{secs // 60}m{secs % 60:02d}s"

    def log_progress(self) -> None:
        logger.info(
            "seen=%d processed=%d skipped=%d upserted=%d actors_enqueued=%d errors=%d elapsed=%s",
            self.records_seen,
            self.records_processed,
            self.records_skipped,
            self.points_upserted,
            self.actors_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


def _safe(val, default: str = "") -> str:
    """Convert a potentially-None field value to a clean string."""
    if val is None:
        return default
    s = str(val).strip()
    return s if s not in ("None", "nan", "NaT") else default


def build_document_text(row: dict) -> str:
    """
    Construct a rich, embeddable incident narrative from the structured row.
    Arranges fields so semantic search captures both the incident context and
    the threat actor profile in a single embedding.
    """
    org = _safe(row.get("affected_organization"))
    country = _safe(row.get("affected_country"))
    industry = _safe(row.get("affected_industry"))
    event_type = _safe(row.get("event_type"))
    event_subtype = _safe(row.get("event_subtype"))
    motive = _safe(row.get("motive"))
    actor = _safe(row.get("actor"))
    actor_type = _safe(row.get("actor_type"))
    actor_country = _safe(row.get("actor_country"))
    description = _safe(row.get("description"))
    event_date = _safe(row.get("event_date"))
    source_url = _safe(row.get("source_url"))

    parts: list[str] = []

    # Header block
    header_parts = []
    if org:
        header_parts.append(f"Target: {org}")
    if country:
        header_parts.append(f"Country: {country}")
    if industry:
        header_parts.append(f"Industry: {industry}")
    if event_date:
        # Truncate to date portion if it's a full timestamp
        date_str = event_date[:10] if len(event_date) > 10 else event_date
        header_parts.append(f"Date: {date_str}")
    if header_parts:
        parts.append("\n".join(header_parts))

    # Attack classification
    attack_parts = []
    if event_type:
        attack_parts.append(f"Attack Type: {event_type}")
    if event_subtype:
        attack_parts.append(f"Attack Method: {event_subtype}")
    if motive:
        attack_parts.append(f"Motive: {motive}")
    if attack_parts:
        parts.append("\n".join(attack_parts))

    # Threat actor
    actor_parts = []
    if actor:
        actor_line = f"Threat Actor: {actor}"
        if actor_type:
            actor_line += f" ({actor_type}"
            if actor_country:
                actor_line += f", {actor_country}"
            actor_line += ")"
        elif actor_country:
            actor_line += f" ({actor_country})"
        actor_parts.append(actor_line)
    if actor_parts:
        parts.append("\n".join(actor_parts))

    # Description (primary intel content)
    if description:
        parts.append(f"Incident Summary:\n{description}")

    # Source reference
    if source_url:
        parts.append(f"Source: {source_url}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class CybersecurityAttacksIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.skip_actors: bool = args.skip_actors
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.min_text_len: int = args.min_text_len

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)

        self._session_seen_actors: set[str] = set()
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

            logger.info(
                "Starting ingestion of %s (limit=%s, resume_from=%d)",
                HF_DATASET_ID,
                limit or "none",
                checkpoint,
            )

            stats = IngestStats()
            await self._ingest(stats, limit, checkpoint)

            await self._flush_upsert_buffer(stats)
            await self._save_checkpoint(stats.records_seen)
            stats.log_progress()
            logger.info("Ingestion complete.")
        finally:
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()

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
    # Dataset ingestion
    # ------------------------------------------------------------------

    async def _ingest(self, stats: IngestStats, limit: int, checkpoint: int) -> None:
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset(HF_DATASET_ID, split="train", streaming=True, token=HF_TOKEN or None)

        async for row in self._stream_rows(ds, stats, limit, checkpoint):
            text = build_document_text(row)
            if len(text) < self.min_text_len:
                stats.records_skipped += 1
                continue

            # Derive a stable deterministic ID from slug or content hash
            slug = _safe(row.get("slug"))
            if slug:
                doc_id = str(uuid.UUID(bytes=hashlib.sha256(slug.encode()).digest()[:16]))
            else:
                doc_id = str(uuid.UUID(bytes=hashlib.sha256(text[:256].encode()).digest()[:16]))

            # Build structured payload metadata
            event_date_raw = _safe(row.get("event_date"))
            metadata: dict = {
                "source": SOURCE_LABEL,
                "dataset": HF_DATASET_ID,
                "doc_id": doc_id,
                "document_type": "cybersecurity_incident",
                "provenance": "public_record",
                "ingest_date": TODAY,
            }
            for field_name, dest_key in [
                ("slug", "slug"),
                ("event_year", "event_year"),
                ("affected_country", "affected_country"),
                ("affected_organization", "affected_organization"),
                ("affected_industry", "affected_industry"),
                ("afftected_industry_code", "industry_code"),  # original typo preserved
                ("event_type", "event_type"),
                ("event_subtype", "event_subtype"),
                ("motive", "motive"),
                ("actor", "actor"),
                ("actor_type", "actor_type"),
                ("actor_country", "actor_country"),
                ("source_url", "source_url"),
            ]:
                val = _safe(row.get(field_name))
                if val:
                    metadata[dest_key] = val
            if event_date_raw:
                metadata["event_date"] = event_date_raw[:10]  # keep date portion only

            try:
                await self._process_record(doc_id, text, metadata, row, stats)
            except Exception as exc:
                stats.errors += 1
                logger.warning("Error processing record %s: %s", doc_id, exc)

        # Save final checkpoint
        await self._save_checkpoint(stats.records_seen)

    async def _process_record(
        self,
        doc_id: str,
        text: str,
        metadata: dict,
        row: dict,
        stats: IngestStats,
    ) -> None:
        stats.records_processed += 1

        # Optionally enqueue threat actor for Cyber desk research
        if not self.skip_actors:
            enqueued = await self._maybe_enqueue_actor(row)
            stats.actors_enqueued += enqueued

        # Build a single PointStruct (incidents are short enough to be one point each)
        point_id = doc_id
        payload = {"text": text, **metadata}
        self._upsert_buffer.append(
            qdrant_models.PointStruct(id=point_id, vector=[0.0] * EMBEDDING_DIM, payload=payload)
        )

        if len(self._upsert_buffer) >= self.upsert_batch_size:
            await self._flush_upsert_buffer(stats)

        if stats.records_processed % 500 == 0:
            stats.log_progress()

    # ------------------------------------------------------------------
    # Threat actor enqueueing
    # ------------------------------------------------------------------

    async def _maybe_enqueue_actor(self, row: dict) -> int:
        """
        Enqueue a Cyber desk research job for known threat actors.
        Only acts on actor types that are meaningful intelligence targets.
        Deduplicates within the session to avoid flooding the queue.
        """
        if not self._redis or self.dry_run:
            return 0

        actor = _safe(row.get("actor"))
        actor_type = _safe(row.get("actor_type"))

        if not actor or actor.lower() in ("unknown", "undetermined", ""):
            return 0
        if actor_type not in RESEARCHED_ACTOR_TYPES:
            return 0

        normalised = actor.lower().strip()
        if normalised in self._session_seen_actors:
            return 0

        # Redis-level dedup
        already_seen = await self._redis.sismember(SEEN_ACTORS_KEY, normalised)
        if already_seen:
            self._session_seen_actors.add(normalised)
            return 0

        actor_country = _safe(row.get("actor_country"))
        topic = actor
        if actor_country:
            topic = f"{actor} ({actor_country})"

        job = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": topic,
                "desk": "cyber-intelligence-and-warfare-desk",
                "priority": "normal",
                "directives_lens": True,
                "triggered_by": "cybersec_attacks_ingest",
            }
        )
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        await self._redis.sadd(SEEN_ACTORS_KEY, normalised)
        self._session_seen_actors.add(normalised)

        logger.info("Research job enqueued: %r → Cyber desk", topic)
        return 1

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
    # Streaming helper
    # ------------------------------------------------------------------

    async def _stream_rows(self, dataset, stats: IngestStats, limit: int, checkpoint: int):
        loop = asyncio.get_event_loop()
        skipped_ff = 0

        def _iter():
            yield from dataset

        it = _iter()

        _sentinel = object()
        _exhausted = False
        while True:
            row = await loop.run_in_executor(None, next, it, _sentinel)
            if row is _sentinel:
                _exhausted = True
                break

            stats.records_seen += 1

            if stats.records_seen <= checkpoint:
                skipped_ff += 1
                if skipped_ff % 1_000 == 0:
                    logger.info("Fast-forwarding checkpoint: %d/%d", skipped_ff, checkpoint)
                continue

            yield row

            if limit and stats.records_seen - checkpoint >= limit:
                logger.info("Reached --limit %d — stopping.", limit)
                break

        if _exhausted:
            logger.info("Dataset exhausted at %d total records — ingestion complete.", stats.records_seen)

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
        description="Ingest vinitvek/cybersecurityattacks dataset into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--limit", type=int, default=0, help="Stop after N source records (0=no limit)")
    p.add_argument("--skip-actors", action="store_true", help="Disable threat actor research job enqueueing")
    p.add_argument("--resume", action="store_true", help="Resume from Redis checkpoint")
    p.add_argument("--dry-run", action="store_true", help="Parse but skip Qdrant writes and Redis updates")
    p.add_argument("--embed-batch-size", type=int, default=48, dest="embed_batch_size")
    p.add_argument("--embed-concurrency", type=int, default=3, dest="embed_concurrency")
    p.add_argument("--upsert-batch-size", type=int, default=64, dest="upsert_batch_size")
    p.add_argument("--min-text-len", type=int, default=40, dest="min_text_len")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not HF_TOKEN:
        parser.error("HF_TOKEN not set in environment — required for dataset access and embeddings.")

    logger.info(
        "Starting cybersecurity attacks ingest | dataset=%s limit=%s skip_actors=%s dry_run=%s",
        HF_DATASET_ID,
        args.limit or "none",
        args.skip_actors,
        args.dry_run,
    )

    if args.dry_run:
        logger.warning("DRY RUN — no data will be written to Qdrant or Redis.")

    ingestor = CybersecurityAttacksIngestor(args)
    asyncio.run(ingestor.run(limit=args.limit, resume=args.resume))


if __name__ == "__main__":
    main()
